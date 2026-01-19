
import kagglehub
from kagglehub import KaggleDatasetAdapter
import pandas as pd
import os

def download_data():
    print("Downloading dataset via kagglehub...")
    
    # Load the dataset using the user's snippet
    # Note: verify if file_path needs to be specific or if empty string loads the main file
    # The snippet had file_path = "", assuming it loads the primary file or all.
    # We will try to load it into a dataframe.
    
    try:
        df = kagglehub.load_dataset(
          KaggleDatasetAdapter.PANDAS,
          "asaniczka/1-3m-linkedin-jobs-and-skills-2024",
          "" # Load all or default
        )
        
        print(f"Dataset loaded. Shape: {df.shape}")
        
        # Ensure data directory exists
        os.makedirs("data", exist_ok=True)
        
        output_path = os.path.join("data", "linkedin_jobs.csv")
        print(f"Saving to {output_path}...")
        df.to_csv(output_path, index=False)
        print("Download and save complete!")
        
    except Exception as e:
        print(f"Error downloading data: {e}")

if __name__ == "__main__":
    download_data()
