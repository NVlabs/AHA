import os
import shutil

def replace_files():
    """
    Replace specific files in target directories with source files.
    """
    # Hardcoded lists of source files and their target directories
    source_files = [
        './aha/Data_Generation/RLBench/rlbench/backend/observation.py',
        './aha/Data_Generation/RLBench/rlbench/backend/scene.py',
        './aha/Data_Generation/RLBench/rlbench/backend/waypoints.py',
        './aha/Data_Generation/RLBench/rlbench/demo.py',
        './aha/Data_Generation/RLBench/rlbench/observation_config.py',
        './aha/Data_Generation/RLBench/rlbench/task_environment.py'
    ]
    
    target_dirs = [
        './RLBench/rlbench/backend',
        './RLBench/rlbench/backend',
        './RLBench/rlbench/backend',
        './RLBench/rlbench',
        './RLBench/rlbench',
        './RLBench/rlbench'
    ]
    
    # Check if the lists have the same length
    if len(source_files) != len(target_dirs):
        raise ValueError("The number of source files must match the number of target directories")
    
    print(f"Starting file replacement for {len(source_files)} files...")
    
    for source_file, target_dir in zip(source_files, target_dirs):
        # Check if source file exists
        if not os.path.isfile(source_file):
            print(f"Warning: Source file '{source_file}' not found. Skipping.")
            continue
        
        # Check if target directory exists
        if not os.path.isdir(target_dir):
            print(f"Warning: Target directory '{target_dir}' not found. Skipping.")
            continue
        
        # Get the filename from the source path
        filename = os.path.basename(source_file)
        
        # Create the target file path
        target_file_path = os.path.join(target_dir, filename)
        
        # Copy the file, overwriting if it exists
        shutil.copy2(source_file, target_file_path)
        print(f"Replaced '{target_file_path}' with '{source_file}'")

def main():
    try:
        replace_files()
        print("File replacement completed successfully.")
    except Exception as e:
        print(f"Error: {e}")
        return 1
    
    return 0

if __name__ == "__main__":
    exit(main())
