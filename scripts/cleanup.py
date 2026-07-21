import os
import shutil
import glob

def cleanup_project():
    print("Starting project cleanup...")
    
    # Define directories
    base_dir = os.path.dirname(os.path.abspath(__file__))
    project_root = os.path.dirname(base_dir)
    output_dir = os.path.join(project_root, 'outputs')
    
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
        print(f"Created output directory: {output_dir}")
    
    # 1. Move Artifacts (*.png, *.pdf, *.html) to outputs/
    patterns = ['*.png', '*.pdf', '*.html']
    moved_count = 0
    
    for pattern in patterns:
        files = glob.glob(os.path.join(project_root, pattern))
        for file_path in files:
            file_name = os.path.basename(file_path)
            # Skip if it's already in the destination (unlikely for root files but good practice)
            dest_path = os.path.join(output_dir, file_name)
            
            try:
                shutil.move(file_path, dest_path)
                print(f"Moved: {file_name} -> outputs/")
                moved_count += 1
            except Exception as e:
                print(f"Error moving {file_name}: {e}")
                
    print(f"Moved {moved_count} artifact files.")

    # 2. Cleanup redundant directories
    dirs_to_remove = [
        'latex_paper_output',
        'research_pdf_output'
    ]
    
    for dir_name in dirs_to_remove:
        dir_path = os.path.join(project_root, dir_name)
        if os.path.exists(dir_path):
            try:
                shutil.rmtree(dir_path)
                print(f"Removed directory: {dir_name}")
            except Exception as e:
                print(f"Error removing {dir_name}: {e}")

    # 3. Consolidate or cleanup redundant scripts is manual, skipping automatic deletion of .py files to avoid accidents.
    
    print("\nCleanup complete!")

if __name__ == "__main__":
    cleanup_project()
