
import kagglehub
import os
import shutil

def download_resume_data():
    print("Downloading resume dataset...")
    try:
        # Download the dataset
        path = kagglehub.dataset_download("mdtalhask/ai-powered-resume-screening-dataset-2025")
        print(f"Resume Dataset downloaded to: {path}")
        
        files = os.listdir(path)
        print(f"Files found: {files}")
        
        csv_file = None
        for f in files:
            if f.endswith(".csv"):
                csv_file = f
                break
        
        if csv_file:
            src = os.path.join(path, csv_file)
            dst = os.path.join("data", "resumes.csv")
            print(f"Copying {src} to {dst}...")
            shutil.copy(src, dst)
            print("Success: Resume file copied.")
        else:
            print("Error: No CSV file found in the downloaded folder.")
            
    except Exception as e:
        print(f"Error during download: {e}")

if __name__ == "__main__":
    download_resume_data()
