import os
import zipfile
import boto3
from dotenv import load_dotenv
import time

# Resolve env file and CloudFormation path relative to this script
script_dir = os.path.dirname(os.path.abspath(__file__))
project_dir = os.path.abspath(os.path.join(script_dir, ".."))

env_path = os.path.join(project_dir, ".env")
if not os.path.exists(env_path):
    env_path = os.path.join(os.getcwd(), ".env")

print(f"Loading env from: {env_path}")
load_dotenv(dotenv_path=env_path)

region = os.getenv('AWS_REGION', 'us-east-1')
bucket = os.getenv('S3_BUCKET_NAME', 'virtual-cmo-raw-signals')
repo_name = "virtual-cmo-backend"
build_project_name = "virtual-cmo-build"
cluster_name = "virtual-cmo-cluster"
service_name = "virtual-cmo-service"

print(f"AWS ECS Fargate Deployer - Region: {region}")

# Setup boto3 clients
ecr_client = boto3.client('ecr', region_name=region)
codebuild_client = boto3.client('codebuild', region_name=region)
ecs_client = boto3.client('ecs', region_name=region)
ec2_client = boto3.client('ec2', region_name=region)
s3_client = boto3.client('s3', region_name=region)
iam_client = boto3.client('iam', region_name=region)

# 1. Zip Code
def create_source_zip():
    zip_path = os.path.join(project_dir, "source.zip")
    print(f"Packaging code into source.zip at {zip_path}...")
    
    ignore_dirs = {'.git', 'agentic_env', 'node_modules', 'frontend', '.next', '__pycache__', 'outputs', 'medical_comparison_results', 'deployment'}
    ignore_files = {'source.zip', '.env', 'agentic_disease_finder.log'}
    
    with zipfile.ZipFile(zip_path, 'w', zipfile.ZIP_DEFLATED) as zipf:
        for root, dirs, files in os.walk(project_dir):
            # Prune directory search
            dirs[:] = [d for d in dirs if d not in ignore_dirs]
            
            for file in files:
                if file in ignore_files or file.endswith(('.png', '.pkl', '.weights.h5', '.h5', '.pt', '.log')):
                    # Skip heavy models, screenshots, and logs
                    continue
                file_path = os.path.join(root, file)
                archive_name = os.path.relpath(file_path, project_dir)
                zipf.write(file_path, archive_name)
                
    print(f"[SUCCESS] Created source.zip ({os.path.getsize(zip_path) / 1024:.1f} KB)")
    return zip_path

# 2. Upload to S3
def upload_to_s3(zip_path):
    s3_key = "builds/source.zip"
    print(f"Uploading source code to S3: s3://{bucket}/{s3_key}...")
    s3_client.upload_file(zip_path, bucket, s3_key)
    print("[SUCCESS] Upload complete!")
    os.remove(zip_path)

# 3. Create Infrastructure via CloudFormation
def deploy_cf_stack(vpc_id, subnets):
    cf_client = boto3.client('cloudformation', region_name=region)
    stack_name = "virtual-cmo-infrastructure"
    
    # Read CF YAML
    cf_template_path = os.path.join(script_dir, "ecs_fargate_cf.yaml")
    with open(cf_template_path, 'r') as f:
        template_body = f.read()
        
    print(f"Deploying CloudFormation stack '{stack_name}' with VPC {vpc_id} and subnets {subnets}...")
    
    params = [
        {'ParameterKey': 'VpcId', 'ParameterValue': vpc_id},
        {'ParameterKey': 'Subnets', 'ParameterValue': ','.join(subnets)}
    ]
    
    try:
        # Check if stack exists
        cf_client.describe_stacks(StackName=stack_name)
        print("Stack exists. Updating stack...")
        try:
            cf_client.update_stack(
                StackName=stack_name,
                TemplateBody=template_body,
                Parameters=params,
                Capabilities=['CAPABILITY_NAMED_IAM']
            )
            print("Waiting for stack update to complete...")
            waiter = cf_client.get_waiter('stack_update_complete')
            waiter.wait(StackName=stack_name)
        except Exception as update_err:
            if "No updates are to be performed" in str(update_err):
                print("No template updates needed.")
            else:
                raise update_err
    except Exception:
        # Create new stack
        cf_client.create_stack(
            StackName=stack_name,
            TemplateBody=template_body,
            Parameters=params,
            Capabilities=['CAPABILITY_NAMED_IAM']
        )
        print("Waiting for stack creation to complete (usually takes 1-2 minutes)...")
        waiter = cf_client.get_waiter('stack_create_complete')
        waiter.wait(StackName=stack_name)
        
    print("[SUCCESS] Infrastructure stack deployed successfully!")
    
    # Get Outputs
    outputs = cf_client.describe_stacks(StackName=stack_name)['Stacks'][0].get('Outputs', [])
    ecr_uri = None
    target_group_arn = None
    api_gateway_endpoint = None
    
    for out in outputs:
        if out['OutputKey'] == 'ECRRepoUri':
            ecr_uri = out['OutputValue']
        elif out['OutputKey'] == 'TargetGroupArn':
            target_group_arn = out['OutputValue']
        elif out['OutputKey'] == 'ApiGatewayEndpoint':
            api_gateway_endpoint = out['OutputValue']
            
    return ecr_uri, target_group_arn, api_gateway_endpoint

# 4. Trigger CodeBuild
def run_codebuild():
    print(f"Triggering CodeBuild project '{build_project_name}' to build Docker image...")
    response = codebuild_client.start_build(projectName=build_project_name)
    build_id = response['build']['id']
    
    print(f"Build started (ID: {build_id}). Monitoring progress...")
    while True:
        status_resp = codebuild_client.batch_get_builds(ids=[build_id])
        build = status_resp['builds'][0]
        status = build['buildStatus']
        print(f"   Current Status: {status}")
        
        if status in ['SUCCEEDED', 'FAILED', 'FAULT', 'STOPPED', 'TIMED_OUT']:
            if status == 'SUCCEEDED':
                print("[SUCCESS] Docker compilation succeeded! ECR image pushed.")
                return True
            else:
                print(f"[ERROR] Docker compilation failed with status: {status}")
                return False
        time.sleep(15)

# 5. Fetch Default VPC and its Public Subnets
def get_vpc_and_subnets():
    vpcs = ec2_client.describe_vpcs(Filters=[{'Name': 'isDefault', 'Values': ['true']}])['Vpcs']
    if not vpcs:
        vpcs = ec2_client.describe_vpcs()['Vpcs']
        
    if not vpcs:
        raise Exception("No VPCs found in this AWS account/region.")
        
    vpc_id = vpcs[0]['VpcId']
    
    # Describe subnets
    sub_resp = ec2_client.describe_subnets(Filters=[{'Name': 'vpc-id', 'Values': [vpc_id]}])['Subnets']
    
    # Filter to subnets that have a route to an internet gateway (igw)
    public_subnets = []
    for subnet in sub_resp:
        subnet_id = subnet['SubnetId']
        rts = ec2_client.describe_route_tables(Filters=[{'Name': 'association.subnet-id', 'Values': [subnet_id]}])['RouteTables']
        if not rts:
            rts = ec2_client.describe_route_tables(Filters=[{'Name': 'vpc-id', 'Values': [vpc_id]}, {'Name': 'association.main', 'Values': ['true']}])['RouteTables']
        
        is_public = False
        for rt in rts:
            for route in rt.get('Routes', []):
                if route.get('GatewayId', '').startswith('igw-'):
                    is_public = True
                    break
        if is_public:
            public_subnets.append(subnet_id)
            
    if not public_subnets:
        public_subnets = [s['SubnetId'] for s in sub_resp]
        
    return vpc_id, public_subnets

# 6. Launch ECS Service
def deploy_ecs_service(target_group_arn, api_gateway_endpoint, vpc_id, subnets):
    print("Launching Fargate Service...")
    print(f"   Using Subnets: {subnets}")
    
    # Get Security Group and Task Definition
    cf_client = boto3.client('cloudformation', region_name=region)
    stack = cf_client.describe_stacks(StackName="virtual-cmo-infrastructure")['Stacks'][0]
    outputs = stack.get('Outputs', [])
    
    security_group_id = None
    for out in outputs:
        if out['OutputKey'] == 'ContainerSecurityGroupId':
            security_group_id = out['OutputValue']
            
    stack_resources = cf_client.describe_stack_resources(StackName="virtual-cmo-infrastructure")['StackResources']
    task_def_arn = None
    for res in stack_resources:
        if res['ResourceType'] == 'AWS::ECS::TaskDefinition':
            task_def_arn = res['PhysicalResourceId']
            
    print(f"   Security Group ID: {security_group_id}")
    print(f"   Task Definition ARN: {task_def_arn}")
    
    # Format env vars for ECS Task
    env_vars = [
        "AWS_ACCESS_KEY_ID", "AWS_SECRET_ACCESS_KEY", "AWS_REGION",
        "S3_BUCKET_NAME", "BEDROCK_MODEL_ID", "PINECONE_API_KEY", "PINECONE_INDEX_NAME",
        "SAGEMAKER_ENDPOINT_BCI2A", "SAGEMAKER_ENDPOINT_EEG_PD", "SAGEMAKER_ENDPOINT_MRI",
        "SAGEMAKER_ENDPOINT_NEUROFORMER", "SAGEMAKER_ENDPOINT_SPECTRA"
    ]
    formatted_env = []
    for var in env_vars:
        val = os.getenv(var)
        if val:
            formatted_env.append({"name": var, "value": val})
    formatted_env.append({"name": "MOCK_AWS", "value": "False"})
    
    # Register updated Task Definition with Environment Variables injected
    base_task_def = ecs_client.describe_task_definition(taskDefinition=task_def_arn)['taskDefinition']
    container_def = base_task_def['containerDefinitions'][0]
    container_def['environment'] = formatted_env
    
    new_task_def = ecs_client.register_task_definition(
        family=base_task_def['family'],
        taskRoleArn=base_task_def['taskRoleArn'],
        executionRoleArn=base_task_def['executionRoleArn'],
        networkMode=base_task_def['networkMode'],
        containerDefinitions=[container_def],
        requiresCompatibilities=base_task_def['requiresCompatibilities'],
        cpu=base_task_def['cpu'],
        memory=base_task_def['memory']
    )['taskDefinition']
    
    new_task_def_arn = new_task_def['taskDefinitionArn']
    print(f"   Registered updated Task Definition: {new_task_def_arn}")

    # Check if ECS service exists
    try:
        service_info = ecs_client.describe_services(cluster=cluster_name, services=[service_name])['services']
        if service_info and service_info[0]['status'] != 'INACTIVE':
            # Check if it has load balancers configured
            has_lb = len(service_info[0].get('loadBalancers', [])) > 0
            if not has_lb:
                print(f"Existing Fargate Service '{service_name}' has no Load Balancer configured. Recreating service to bind to ALB Target Group...")
                # Scale down
                print("Scaling down existing service to 0...")
                ecs_client.update_service(
                    cluster=cluster_name,
                    service=service_name,
                    desiredCount=0
                )
                print("Deleting service...")
                ecs_client.delete_service(
                    cluster=cluster_name,
                    service=service_name
                )
                # Wait for service to be fully deleted
                print("Waiting for service deletion to complete...")
                while True:
                    check = ecs_client.describe_services(cluster=cluster_name, services=[service_name])['services']
                    if not check or check[0]['status'] == 'INACTIVE':
                        print("Service deleted successfully.")
                        break
                    print("   Waiting for active tasks to terminate and service status to become INACTIVE...")
                    time.sleep(10)
                # Force recreation
                raise Exception("Recreate Service")
            else:
                print(f"Updating existing Fargate Service '{service_name}'...")
                ecs_client.update_service(
                    cluster=cluster_name,
                    service=service_name,
                    taskDefinition=new_task_def_arn,
                    forceNewDeployment=True
                )
        else:
            raise Exception("Service not active")
    except Exception as e:
        if str(e) != "Recreate Service" and "Service not active" not in str(e):
            print(f"Error checking service: {e}. Attempting service creation...")
            
        print(f"Creating new Fargate Service '{service_name}' with load balancer configurations...")
        ecs_client.create_service(
            cluster=cluster_name,
            serviceName=service_name,
            taskDefinition=new_task_def_arn,
            launchType='FARGATE',
            desiredCount=1,
            networkConfiguration={
                'awsvpcConfiguration': {
                    'subnets': subnets,
                    'securityGroups': [security_group_id],
                    'assignPublicIp': 'ENABLED'
                }
            },
            loadBalancers=[
                {
                    'targetGroupArn': target_group_arn,
                    'containerName': 'backend',
                    'containerPort': 8000
                }
            ]
        )
        
    print("[SUCCESS] ECS Service deployment triggered.")
    print("Waiting for Task to provision and start...")
    time.sleep(20)
    
    print(f"\n[SUCCESS] DEPLOYMENT SUCCESSFUL!")
    print(f"   FastAPI Backend HTTPS API Gateway Endpoint: {api_gateway_endpoint}")
    print(f"   Use this URL on Vercel as NEXT_PUBLIC_API_URL!")

# Run Pipeline
if __name__ == "__main__":
    vpc_id, subnets = get_vpc_and_subnets()
    print(f"Detected VPC: {vpc_id}")
    print(f"Detected Public Subnets: {subnets}")
    
    zip_file = create_source_zip()
    upload_to_s3(zip_file)
    
    ecr_uri, target_group_arn, api_gateway_endpoint = deploy_cf_stack(vpc_id, subnets)
    print(f"ECR Repo URI: {ecr_uri}")
    print(f"Target Group ARN: {target_group_arn}")
    print(f"API Gateway Endpoint: {api_gateway_endpoint}")
    
    if run_codebuild():
        deploy_ecs_service(target_group_arn, api_gateway_endpoint, vpc_id, subnets)
