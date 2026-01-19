
import kagglehub
import os
import shutil

def download_data_debug():
    print("Attempting to download dataset via kagglehub.dataset_download...")
    try:
        # Download the dataset files
        path = kagglehub.dataset_download("asaniczka/1-3m-linkedin-jobs-and-skills-2024")
        print(f"Dataset downloaded to: {path}")
        
        # List files in the directory
        files = os.listdir(path)
        print(f"Files found: {files}")
        
        # Identify the CSV file
        csv_file = None
        for f in files:
            if f.endswith(".csv"):
                csv_file = f
                break
        
        if csv_file:
            src = os.path.join(path, csv_file)
            dst = os.path.join("data", "linkedin_jobs.csv")
            print(f"Copying {src} to {dst}...")
            shutil.copy(src, dst)
            print("Success: File copied.")
        else:
            print("Error: No CSV file found in the downloaded folder.")
            
    except Exception as e:
        print(f"Error during download: {e}")

if __name__ == "__main__":
    download_data_debug()
