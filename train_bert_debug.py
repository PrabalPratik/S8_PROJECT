
import pandas as pd
import torch
import torch.nn as nn
from models.bert_ranker import BertRanker
from torch.utils.data import DataLoader, Dataset
from tqdm import tqdm
import os

class ResumeDataset(Dataset):
    def __init__(self, csv_path):
        self.data = pd.read_csv(csv_path)
        # Filter for rows that have necessary columns
        self.data = self.data.dropna(subset=['Skills', 'Job Role', 'Recruiter Decision'])
        
    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, idx):
        row = self.data.iloc[idx]
        
        # Construct Resume Text Representation
        resume_text = f"Skills: {row['Skills']}. Experience: {row['Experience (Years)']} years. Education: {row['Education']}."
        
        # Construct Query (JD) Text Representation - Using Role as proxy for JD
        jd_text = f"Job Role: {row['Job Role']}"
        
        # Label: Hire = 1.0, else 0.0
        label = 1.0 if str(row['Recruiter Decision']).strip().lower() == 'hire' else 0.0
        
        return resume_text, jd_text, label

def train_bert_debug():
    print("Initializing BERT Ranker Debug Training...")
    
    # Settings
    csv_path = "data/resumes.csv"
    model_name = "distilbert-base-uncased" # Faster for debug
    batch_size = 4
    epochs = 3
    lr = 2e-5
    
    # Initialize Model
    # Note: bert_ranker.py uses 'bert-base-uncased' default, but AutoModel handles distilbert fine mostly, 
    # except 'token_type_ids' which distilbert doesn't use. 
    # Let's stick to 'bert-base-uncased' to avoid architecture mismatch in the forward pass if not handled.
    model_name = "bert-base-uncased" 
    
    ranker = BertRanker(model_name=model_name)
    optimizer = torch.optim.AdamW(ranker.parameters(), lr=lr)
    criterion = nn.BCELoss()
    
    # Dataset
    dataset = ResumeDataset(csv_path)
    # Use subset for debug speed
    dataset.data = dataset.data.iloc[:40] 
    print(f"Dataset subset size: {len(dataset)}")
    
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True)
    
    # Training Loop
    print(f"Starting training for {epochs} epochs...")
    for epoch in range(epochs):
        ranker.train()
        total_loss = 0
        
        loop = tqdm(dataloader, desc=f"Epoch {epoch+1}")
        for resume_text, jd_text, label in loop:
            # Training step
            # Resume_text/jd_text are tuples from DataLoader, convert to list
            loss = ranker.train_step(list(resume_text), list(jd_text), label, optimizer, criterion)
            
            total_loss += loss
            loop.set_postfix(loss=loss)
            
    print("Training complete.")
    
    # Test Prediction
    print("\nTesting Prediction on sample...")
    sample_resume = "Skills: Python, Machine Learning, PyTorch. Experience: 5 years."
    sample_jd = "Job Role: AI Researcher"
    
    score = ranker.predict(sample_resume, sample_jd)
    print(f"Resume: {sample_resume}")
    print(f"Target Role: {sample_jd}")
    print(f"Predicted Relevance Score: {score:.4f}")

if __name__ == "__main__":
    train_bert_debug()
