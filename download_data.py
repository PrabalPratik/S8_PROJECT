
from datasets import load_dataset
import pandas as pd
import os

def download_data():
    print("Downloading dataset batuhanmtl/job-skill-set...")
    # Load dataset
    ds = load_dataset("batuhanmtl/job-skill-set")
    
    # Convert to pandas dataframe (usually 'train' split contains the data)
    df = ds['train'].to_pandas()
    
    output_path = os.path.join("data", "job_skills.csv")
    print(f"Saving to {output_path}...")
    df.to_csv(output_path, index=False)
    print("Download complete!")

if __name__ == "__main__":
    download_data()
