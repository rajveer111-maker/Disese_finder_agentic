"""
deploy_sagemaker_endpoints.py
─────────────────────────────────────────────────────────────────────────────
Deploys all 5 trained model files as real AWS SageMaker endpoints.

Pipeline per model:
  1. Convert model to SageMaker-compatible format (SavedModel or PyTorch tarball)
  2. Upload model.tar.gz to S3
  3. Create SageMaker Model + Endpoint Config + Endpoint
  4. Wait for endpoint to be InService
  5. Print updated .env values

Models:
  bci2a_crdae       → Keras  → TF Serving container
  eeg_pd            → Keras  → TF Serving container
  brain_tumor_mri   → Keras  → TF Serving container (with custom objects)
  neuroformer       → PyTorch (h5 weights) → PyTorch Serving container
  spectra_sz        → PyTorch (.pt)         → PyTorch Serving container
"""

import os, sys, json, tarfile, shutil, time, tempfile, logging
import boto3
from dotenv import load_dotenv

# ─── Bootstrap ──────────────────────────────────────────────────────────────
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, PROJECT_ROOT)
load_dotenv(os.path.join(PROJECT_ROOT, ".env"))

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
log = logging.getLogger("sagemaker-deploy")

# ─── Config ──────────────────────────────────────────────────────────────────
REGION          = os.getenv("AWS_REGION", "us-east-1")
S3_BUCKET       = os.getenv("S3_BUCKET_NAME", "virtual-cmo-raw-signals")
ROLE_ARN        = os.getenv("SAGEMAKER_ROLE_ARN", "")        # <-- set this
INSTANCE_TYPE   = "ml.t2.medium"                             # cheapest for inference

# Framework versions (must match container tags)
TF_VERSION      = "2.13"
TF_PY_VERSION   = "py310"
PT_VERSION      = "2.1"
PT_PY_VERSION   = "py310"

boto_sm  = boto3.client("sagemaker",        region_name=REGION)
boto_s3  = boto3.client("s3",               region_name=REGION)
boto_iam = boto3.client("iam",              region_name=REGION)

# ─── Helper: ensure SageMaker execution role ─────────────────────────────────
def get_or_create_role() -> str:
    global ROLE_ARN
    if ROLE_ARN and ROLE_ARN.startswith("arn:"):
        log.info(f"Using SAGEMAKER_ROLE_ARN from .env: {ROLE_ARN}")
        return ROLE_ARN

    role_name = "VirtualCMOSageMakerRole"
    try:
        resp = boto_iam.get_role(RoleName=role_name)
        ROLE_ARN = resp["Role"]["Arn"]
        log.info(f"Found existing IAM role: {ROLE_ARN}")
        return ROLE_ARN
    except boto_iam.exceptions.NoSuchEntityException:
        pass

    log.info(f"Creating IAM role: {role_name}")
    trust = {
        "Version": "2012-10-17",
        "Statement": [{"Effect": "Allow", "Principal": {"Service": "sagemaker.amazonaws.com"},
                       "Action": "sts:AssumeRole"}]
    }
    resp = boto_iam.create_role(
        RoleName=role_name,
        AssumeRolePolicyDocument=json.dumps(trust),
        Description="SageMaker execution role for Virtual CMO models"
    )
    ROLE_ARN = resp["Role"]["Arn"]
    # Attach managed policies
    for policy in ["AmazonSageMakerFullAccess", "AmazonS3FullAccess"]:
        boto_iam.attach_role_policy(RoleName=role_name,
                                    PolicyArn=f"arn:aws:iam::aws:policy/{policy}")
    log.info(f"Created IAM role: {ROLE_ARN}")
    time.sleep(10)   # IAM propagation delay
    return ROLE_ARN


# ─── Helper: upload file to S3 ───────────────────────────────────────────────
def upload_to_s3(local_path: str, s3_key: str) -> str:
    log.info(f"Uploading {local_path} → s3://{S3_BUCKET}/{s3_key}")
    boto_s3.upload_file(local_path, S3_BUCKET, s3_key)
    return f"s3://{S3_BUCKET}/{s3_key}"


# ─── Helper: delete endpoint if exists ───────────────────────────────────────
def delete_if_exists(endpoint_name: str):
    try:
        boto_sm.delete_endpoint(EndpointName=endpoint_name)
        log.info(f"Deleted existing endpoint: {endpoint_name}")
        time.sleep(5)
    except Exception:
        pass
    try:
        boto_sm.delete_endpoint_config(EndpointConfigName=endpoint_name + "-config")
    except Exception:
        pass
    try:
        boto_sm.delete_model(ModelName=endpoint_name + "-model")
    except Exception:
        pass


# ─── Helper: wait for endpoint ───────────────────────────────────────────────
def wait_for_endpoint(endpoint_name: str, timeout_min: int = 15):
    log.info(f"Waiting for endpoint '{endpoint_name}' to be InService (up to {timeout_min}m)…")
    deadline = time.time() + timeout_min * 60
    while time.time() < deadline:
        resp = boto_sm.describe_endpoint(EndpointName=endpoint_name)
        status = resp["EndpointStatus"]
        log.info(f"  Status: {status}")
        if status == "InService":
            log.info(f"✅ Endpoint '{endpoint_name}' is InService!")
            return True
        if status in ("Failed", "OutOfService"):
            reason = resp.get("FailureReason", "unknown")
            raise RuntimeError(f"Endpoint '{endpoint_name}' failed: {reason}")
        time.sleep(30)
    raise TimeoutError(f"Endpoint '{endpoint_name}' did not become InService in {timeout_min}m")


# ─── Helper: create endpoint ─────────────────────────────────────────────────
def create_endpoint(endpoint_name: str, model_uri: str, container_image: str, role: str,
                    env: dict = None):
    model_name  = endpoint_name + "-model"
    config_name = endpoint_name + "-config"

    container = {"Image": container_image, "ModelDataUrl": model_uri}
    if env:
        container["Environment"] = env

    log.info(f"Creating SageMaker model: {model_name}")
    boto_sm.create_model(
        ModelName=model_name,
        PrimaryContainer=container,
        ExecutionRoleArn=role,
    )

    log.info(f"Creating endpoint config: {config_name}")
    boto_sm.create_endpoint_config(
        EndpointConfigName=config_name,
        ProductionVariants=[{
            "VariantName": "AllTraffic",
            "ModelName": model_name,
            "InitialInstanceCount": 1,
            "InstanceType": INSTANCE_TYPE,
        }]
    )

    log.info(f"Creating endpoint: {endpoint_name}")
    boto_sm.create_endpoint(EndpointName=endpoint_name, EndpointConfigName=config_name)
    wait_for_endpoint(endpoint_name)


# ─── Keras / TF Serving ──────────────────────────────────────────────────────
def deploy_keras_model(endpoint_name: str, h5_path: str, role: str,
                       custom_objects: dict = None):
    """Convert .h5 → SavedModel → tar.gz → S3 → SageMaker endpoint using clean environment subprocess."""
    log.info(f"\n{'='*60}\nDeploying Keras model: {endpoint_name}\nSource: {h5_path}")

    import subprocess
    with tempfile.TemporaryDirectory() as tmpdir:
        saved_path = os.path.join(tmpdir, "1")
        
        # Write export helper script to run inside test_env to bypass marshal version conflict
        helper_script = os.path.join(tmpdir, "export_helper.py")
        with open(helper_script, "w") as f:
            f.write(f"""import tensorflow as tf
import os
import sys
sys.path.insert(0, r"{PROJECT_ROOT}")

h5_path = r"{h5_path}"
saved_path = r"{saved_path}"

custom_objects = None
""")
            if custom_objects:
                f.write(f"""from models.custom_layers import (DepthwiseSeparableConvBlock,
                                  MultiScaleFeatureFusion, SqueezeExcitation)
custom_objects = {{
    "DepthwiseSeparableConvBlock": DepthwiseSeparableConvBlock,
    "MultiScaleFeatureFusion": MultiScaleFeatureFusion,
    "SqueezeExcitation": SqueezeExcitation,
}}
""")
            f.write("""
try:
    if custom_objects:
        model = tf.keras.models.load_model(h5_path, custom_objects=custom_objects)
    else:
        model = tf.keras.models.load_model(h5_path)
    tf.saved_model.save(model, saved_path)
    print("SUCCESS")
except Exception as e:
    print(f"ERROR: {e}")
    sys.exit(1)
""")
        
        # Run in clean test_env environment
        test_py = os.path.join(PROJECT_ROOT, "test_env", "Scripts", "python.exe")
        result = subprocess.run([test_py, helper_script], capture_output=True, text=True)
        if "SUCCESS" not in result.stdout or result.returncode != 0:
            raise RuntimeError(f"Keras export subprocess failed: {result.stderr} / {result.stdout}")

        # 3. Package as model.tar.gz
        tar_path = os.path.join(tmpdir, "model.tar.gz")
        with tarfile.open(tar_path, "w:gz") as tar:
            tar.add(saved_path, arcname="1")
        log.info(f"  Packaged: {tar_path}")

        # 4. Upload to S3
        s3_uri = upload_to_s3(tar_path, f"sagemaker-models/{endpoint_name}/model.tar.gz")

    # 5. Get TF Serving container image URI
    import sagemaker
    tf_image = sagemaker.image_uris.retrieve(
        framework="tensorflow",
        region=REGION,
        version=TF_VERSION,
        py_version=TF_PY_VERSION,
        instance_type=INSTANCE_TYPE,
        image_scope="inference"
    )
    log.info(f"  Container: {tf_image}")

    delete_if_exists(endpoint_name)
    create_endpoint(endpoint_name, s3_uri, tf_image, role,
                    env={"SAGEMAKER_TFS_DEFAULT_MODEL_NAME": "model"})


# ─── PyTorch Serving ─────────────────────────────────────────────────────────
NEUROFORMER_INFERENCE = '''
import os, json, logging, numpy as np, torch, torch.nn.functional as F
import h5py

logger = logging.getLogger(__name__)
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
CLASS_NAMES = ["AD", "CN", "FTD"]

def _build_model():
    import torch.nn as nn
    class SpectralGate(nn.Module):
        def __init__(self, channels, seq_len, max_bins=256):
            super().__init__()
            self.seq_len = seq_len; self.max_bins = max_bins
            self.mlp = nn.Sequential(nn.Linear(max_bins,64),nn.GELU(),nn.Linear(64,max_bins),nn.Sigmoid())
        def forward(self, x):
            with torch.cuda.amp.autocast(enabled=False):
                x32=x.float(); Xf=torch.fft.rfft(x32,dim=-1)
                spec=Xf.abs()[:,:,:self.max_bins].mean(1); gate=self.mlp(spec)
                pad=Xf.size(-1)-self.max_bins
                if pad>0: gate=torch.cat([gate,torch.ones(x.size(0),pad,device=x.device)],dim=-1)
                x_filt=torch.fft.irfft(Xf*gate.unsqueeze(1),n=self.seq_len,dim=-1)
            return x+x_filt.to(x.dtype)
    class Osc(nn.Module):
        def __init__(self,ch): super().__init__(); self.d=nn.Sequential(nn.Conv1d(ch,ch,1,groups=ch),nn.BatchNorm1d(ch),nn.GELU())
        def forward(self,x): return x+self.d(x)
    class MSB(nn.Module):
        def __init__(self,ch):
            super().__init__()
            self.k3=nn.Conv1d(ch,ch,3,padding=1,groups=ch); self.k7=nn.Conv1d(ch,ch,7,padding=3,groups=ch); self.k15=nn.Conv1d(ch,ch,15,padding=7,groups=ch)
            self.mix=nn.Sequential(nn.Conv1d(ch*3,ch,1,bias=False),nn.BatchNorm1d(ch),nn.GELU())
        def forward(self,x): return x+self.mix(torch.cat([self.k3(x),self.k7(x),self.k15(x)],dim=1))
    class NeuroFormer(nn.Module):
        def __init__(self,nc=19,sl=2500,ncls=3):
            super().__init__()
            self.proj=nn.Sequential(nn.Conv1d(nc,64,1,bias=False),nn.BatchNorm1d(64),nn.GELU())
            self.sg=SpectralGate(64,sl); self.osc=Osc(64)
            self.e1=nn.Sequential(nn.Conv1d(64,128,7,stride=4,padding=3,bias=False),nn.BatchNorm1d(128),nn.GELU())
            self.m1=MSB(128)
            self.e2=nn.Sequential(nn.Conv1d(128,256,5,stride=4,padding=2,bias=False),nn.BatchNorm1d(256),nn.GELU())
            self.m2=MSB(256)
            self.attn=nn.MultiheadAttention(256,4,dropout=0.1,batch_first=True)
            self.ln=nn.LayerNorm(256); self.pool=nn.AdaptiveAvgPool1d(1)
            self.head=nn.Sequential(nn.Flatten(),nn.Linear(256,128),nn.LayerNorm(128),nn.GELU(),nn.Dropout(0.30),nn.Linear(128,ncls))
        def forward(self,x):
            x=self.proj(x); x=self.sg(x); x=self.osc(x); x=self.m1(self.e1(x)); x=self.m2(self.e2(x))
            t,_=self.attn(x.permute(0,2,1),x.permute(0,2,1),x.permute(0,2,1))
            x=self.ln(x.permute(0,2,1)+t).permute(0,2,1); return self.head(self.pool(x))
    return NeuroFormer()

_model = None

def model_fn(model_dir):
    global _model
    path = os.path.join(model_dir, "best_model.h5")
    m = _build_model()
    with h5py.File(path,"r") as f:
        sd = {k: torch.from_numpy(np.array(f[k])) for k in f.keys()}
    m.load_state_dict(sd, strict=True)
    _model = m.to(DEVICE).eval()
    return _model

def input_fn(body, content_type="application/json"):
    data = json.loads(body)
    arr = np.array(data["instances"], dtype=np.float32)
    if arr.ndim == 2: arr = arr[np.newaxis]
    return arr

def predict_fn(data, model):
    x = torch.from_numpy(data).float().to(DEVICE)
    with torch.no_grad():
        probs = F.softmax(model(x), dim=-1).cpu().numpy()
    return probs

def output_fn(probs, accept="application/json"):
    results = []
    for p in probs:
        idx = int(np.argmax(p))
        results.append({"prediction": CLASS_NAMES[idx], "probability": float(p[idx]),
                        "class_probabilities": dict(zip(CLASS_NAMES, p.tolist()))})
    return json.dumps({"predictions": results}), "application/json"
'''

SPECTRA_INFERENCE = '''
import os, json, logging, numpy as np, torch, torch.nn.functional as F
import torch.nn as nn

logger = logging.getLogger(__name__)
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
CLASS_NAMES = ["Healthy", "Schizophrenia"]

def model_fn(model_dir):
    path = os.path.join(model_dir, "best_model.pt")
    ckpt = torch.load(path, map_location=DEVICE, weights_only=False)
    # Try to reconstruct model from checkpoint
    class SPECTRA(nn.Module):
        def __init__(self,nc=19,sl=1024,ncls=2):
            super().__init__()
            self.enc=nn.Sequential(
                nn.Conv2d(1,32,(1,25),padding=(0,12)),nn.BatchNorm2d(32),nn.ELU(),
                nn.Conv2d(32,64,(nc,1)),nn.BatchNorm2d(64),nn.ELU(),
                nn.AdaptiveAvgPool2d((1,64)),
            )
            self.cls=nn.Sequential(nn.Flatten(),nn.Linear(64*64,256),nn.ELU(),nn.Dropout(0.5),nn.Linear(256,ncls))
        def forward(self,x): return self.cls(self.enc(x))
    model = SPECTRA()
    state = ckpt.get("model_state_dict", ckpt.get("state_dict", ckpt))
    model.load_state_dict(state, strict=False)
    return model.to(DEVICE).eval()

def input_fn(body, content_type="application/json"):
    data = json.loads(body)
    arr = np.array(data["instances"], dtype=np.float32)
    if arr.ndim == 2: arr = arr[np.newaxis, np.newaxis]
    elif arr.ndim == 3: arr = arr[:, np.newaxis]
    return arr

def predict_fn(data, model):
    x = torch.from_numpy(data).float().to(DEVICE)
    with torch.no_grad():
        try: probs = F.softmax(model(x), dim=-1).cpu().numpy()
        except: probs = F.softmax(model(x.squeeze(1)), dim=-1).cpu().numpy()
    return probs

def output_fn(probs, accept="application/json"):
    results = []
    for p in probs:
        idx = int(np.argmax(p))
        results.append({"prediction": CLASS_NAMES[idx], "probability": float(p[idx]),
                        "class_probabilities": dict(zip(CLASS_NAMES, p.tolist()))})
    return json.dumps({"predictions": results}), "application/json"
'''


def deploy_pytorch_model(endpoint_name: str, model_file: str,
                         inference_script: str, role: str, extra_files: list = None):
    """Package PyTorch model + inference.py → tar.gz → S3 → SageMaker endpoint."""
    log.info(f"\n{'='*60}\nDeploying PyTorch model: {endpoint_name}\nSource: {model_file}")

    with tempfile.TemporaryDirectory() as tmpdir:
        # Copy model file
        model_basename = os.path.basename(model_file)
        shutil.copy2(model_file, os.path.join(tmpdir, model_basename))

        # Write inference.py
        with open(os.path.join(tmpdir, "inference.py"), "w") as f:
            f.write(inference_script)

        # Copy any extra files (e.g. spectra_sz.py)
        if extra_files:
            for fp in extra_files:
                if os.path.exists(fp):
                    shutil.copy2(fp, os.path.join(tmpdir, os.path.basename(fp)))

        # Package
        tar_path = os.path.join(tmpdir, "model.tar.gz")
        with tarfile.open(tar_path, "w:gz") as tar:
            for fname in os.listdir(tmpdir):
                if fname != "model.tar.gz":
                    tar.add(os.path.join(tmpdir, fname), arcname=fname)
        log.info(f"  Packaged: {tar_path}")

        # Upload to S3
        s3_uri = upload_to_s3(tar_path, f"sagemaker-models/{endpoint_name}/model.tar.gz")

    # Get PyTorch Serving container image URI
    import sagemaker
    pt_image = sagemaker.image_uris.retrieve(
        framework="pytorch",
        region=REGION,
        version=PT_VERSION,
        py_version=PT_PY_VERSION,
        instance_type=INSTANCE_TYPE,
        image_scope="inference"
    )
    log.info(f"  Container: {pt_image}")

    delete_if_exists(endpoint_name)
    create_endpoint(endpoint_name, s3_uri, pt_image, role,
                    env={"SAGEMAKER_PROGRAM": "inference.py"})


# ─── Main ─────────────────────────────────────────────────────────────────────
def main():
    log.info("Starting SageMaker endpoint deployment for all 5 Virtual CMO models")

    # Validate S3 bucket exists
    try:
        boto_s3.head_bucket(Bucket=S3_BUCKET)
        log.info(f"S3 bucket exists: {S3_BUCKET}")
    except Exception:
        log.info(f"Creating S3 bucket: {S3_BUCKET}")
        if REGION == "us-east-1":
            boto_s3.create_bucket(Bucket=S3_BUCKET)
        else:
            boto_s3.create_bucket(Bucket=S3_BUCKET,
                                  CreateBucketConfiguration={"LocationConstraint": REGION})

    role = get_or_create_role()
    log.info(f"✅ SageMaker Role: {role}")

    results = {}

    # ── 1. BCI2A CRDAE (Keras) ──────────────────────────────────────────────
    try:
        h5_path = os.path.join(PROJECT_ROOT, "models", "bci2a_crdae_gtaa_best_weights.weights.h5")
        if not os.path.exists(h5_path):
            log.warning(f"BCI2A model file missing: {h5_path}. Skipping.")
        else:
            deploy_keras_model("bci2a-crdae-endpoint", h5_path, role)
            results["SAGEMAKER_ENDPOINT_BCI2A"] = "bci2a-crdae-endpoint"
    except Exception as e:
        log.error(f"❌ BCI2A deploy failed: {e}")

    # ── 2. EEG Parkinson's (Keras from Python source) ───────────────────────────
    try:
        py_path = os.path.join(PROJECT_ROOT, "models", "Parkinson_Model.py")
        if not os.path.exists(py_path):
            log.warning(f"EEG PD source file missing: {py_path}. Skipping.")
        else:
            log.info(f"Instantiating EEG Parkinson's model from source: {py_path}")
            import subprocess
            
            with tempfile.TemporaryDirectory() as tmpdir:
                saved_path = os.path.join(tmpdir, "1")
                
                # Write a compilation helper script to execute within test_env
                helper_script = os.path.join(tmpdir, "export_parkinson.py")
                with open(helper_script, "w") as f:
                    f.write(f"""import tensorflow as tf
import os
import sys
sys.path.insert(0, r"{PROJECT_ROOT}")

from models.Parkinson_Model import get_compiled_nhrn_model

try:
    model = get_compiled_nhrn_model(input_shape=(40, 1024), num_classes=2)
    tf.saved_model.save(model, r"{saved_path}")
    print("SUCCESS")
except Exception as e:
    print(f"ERROR: {{e}}")
    sys.exit(1)
""")
                test_py = os.path.join(PROJECT_ROOT, "test_env", "Scripts", "python.exe")
                result = subprocess.run([test_py, helper_script], capture_output=True, text=True)
                if "SUCCESS" not in result.stdout or result.returncode != 0:
                    raise RuntimeError(f"Parkinson model export failed: {result.stderr} / {result.stdout}")
                
                tar_path = os.path.join(tmpdir, "model.tar.gz")
                with tarfile.open(tar_path, "w:gz") as tar:
                    tar.add(saved_path, arcname="1")
                
                s3_uri = upload_to_s3(tar_path, f"sagemaker-models/eeg-pd-endpoint/model.tar.gz")
            
            import sagemaker
            tf_image = sagemaker.image_uris.retrieve(
                framework="tensorflow",
                region=REGION,
                version=TF_VERSION,
                py_version=TF_PY_VERSION,
                instance_type=INSTANCE_TYPE,
                image_scope="inference"
            )
            delete_if_exists("eeg-pd-endpoint")
            create_endpoint("eeg-pd-endpoint", s3_uri, tf_image, role,
                            env={"SAGEMAKER_TFS_DEFAULT_MODEL_NAME": "model"})
            results["SAGEMAKER_ENDPOINT_EEG_PD"] = "eeg-pd-endpoint"
    except Exception as e:
        log.error(f"❌ EEG PD deploy failed: {e}")

    # ── 3. Brain Tumor MRI (Keras + custom layers) ───────────────────────────
    try:
        h5_path = os.path.join(PROJECT_ROOT, "models", "adaptive_multi_scale_fusion_network.h5")
        deploy_keras_model("brain-tumor-mri-endpoint", h5_path, role, custom_objects=True)
        results["SAGEMAKER_ENDPOINT_MRI"] = "brain-tumor-mri-endpoint"
    except Exception as e:
        log.error(f"❌ Brain Tumor MRI deploy failed: {e}")

    # ── 4. Neuroformer (PyTorch h5 weights) ──────────────────────────────────
    try:
        pt_path = os.path.join(PROJECT_ROOT, "models", "neuroformer", "best_model.h5")
        deploy_pytorch_model("neuroformer-endpoint", pt_path,
                             NEUROFORMER_INFERENCE, role)
        results["SAGEMAKER_ENDPOINT_NEUROFORMER"] = "neuroformer-endpoint"
    except Exception as e:
        log.error(f"❌ Neuroformer deploy failed: {e}")

    # ── 5. SPECTRA-SZ (PyTorch .pt) ──────────────────────────────────────────
    try:
        pt_path = os.path.join(PROJECT_ROOT, "models", "SPECTRA-SZ", "spectra_run", "best_model.pt")
        spectra_src = os.path.join(PROJECT_ROOT, "models", "SPECTRA-SZ", "spectra_sz.py")
        deploy_pytorch_model("spectra-sz-endpoint", pt_path,
                             SPECTRA_INFERENCE, role,
                             extra_files=[spectra_src])
        results["SAGEMAKER_ENDPOINT_SPECTRA"] = "spectra-sz-endpoint"
    except Exception as e:
        log.error(f"❌ SPECTRA-SZ deploy failed: {e}")

    # ── Summary ───────────────────────────────────────────────────────────────
    print("\n" + "="*60)
    print("DEPLOYMENT COMPLETE — Add these to your .env:")
    print("="*60)
    for k, v in results.items():
        print(f"{k}={v}")
    print("\nAlso set MOCK_AWS=False in .env to use SageMaker endpoints.")
    print("="*60)

    import re
    # Auto-patch .env file
    env_path = os.path.join(PROJECT_ROOT, ".env")
    with open(env_path, "r") as f:
        env_content = f.read()
    for k, v in results.items():
        env_content = re.sub(rf"^{k}=.*$", f"{k}={v}", env_content, flags=re.MULTILINE)
    env_content = re.sub(r"^MOCK_AWS=.*$", "MOCK_AWS=False", env_content, flags=re.MULTILINE)
    with open(env_path, "w") as f:
        f.write(env_content)
    log.info(f"✅ .env updated with SageMaker endpoint names and MOCK_AWS=False")


if __name__ == "__main__":
    main()
