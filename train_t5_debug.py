
import torch
from transformers import T5Tokenizer
from models.t5_generator import T5JDGenerator
from data.dataset_loader import JDDataset
from torch.utils.data import DataLoader

def test_training():
    print("Initializing T5 Debug Training...")
    
    # Paths
    data_path = "data/job_skills.csv"
    model_name = "t5-small"
    
    # Initialize Model & Tokenizer
    print(f"Loading model: {model_name}")
    t5_model = T5JDGenerator(model_name)
    tokenizer = T5Tokenizer.from_pretrained(model_name)
    
    # Initialize Dataset
    print(f"Loading dataset from {data_path}")
    dataset = JDDataset(data_path, tokenizer, max_length=128) # Short length for debug
    
    # Subset for quick text
    dataset.data = dataset.data.iloc[:50] 
    print(f"Dataset size: {len(dataset)}")
    
    # Run a few steps
    print("Starting training loop (2 epochs debug)...")
    # Note: t5_model.train creates its own DataLoader, so we pass the dataset directly
    t5_model.train(dataset, epochs=2, batch_size=4) 

    
    print("Training debug complete.")
    
    # specific test for generation
    print("Testing Generation...")
    sample_role = "Software Engineer"
    sample_skills = "Python, PyTorch, SQL"
    generated_jd = t5_model.generate(sample_role, sample_skills)
    print(f"Input: Role={sample_role}, Skills={sample_skills}")
    print(f"Generated Output: {generated_jd}")

if __name__ == "__main__":
    test_training()
