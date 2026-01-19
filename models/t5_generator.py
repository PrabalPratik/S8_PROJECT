
from transformers import T5ForConditionalGeneration, T5Tokenizer
import torch
from torch.utils.data import DataLoader
from tqdm import tqdm
import os

class T5JDGenerator:
    def __init__(self, model_name="t5-small", device=None):
        self.device = device if device else ("cuda" if torch.cuda.is_available() else "cpu")
        print(f"Loading T5 model on {self.device}...")
        self.tokenizer = T5Tokenizer.from_pretrained(model_name)
        self.model = T5ForConditionalGeneration.from_pretrained(model_name).to(self.device)
        
    def train(self, dataset, output_dir="models/saved_t5", epochs=1, batch_size=4, lr=1e-4):
        dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True)
        optimizer = torch.optim.AdamW(self.model.parameters(), lr=lr)
        
        self.model.train()
        for epoch in range(epochs):
            loop = tqdm(dataloader, desc=f"Epoch {epoch+1}")
            for batch in loop:
                input_ids = batch["input_ids"].to(self.device)
                attention_mask = batch["attention_mask"].to(self.device)
                labels = batch["labels"].to(self.device)
                
                outputs = self.model(
                    input_ids=input_ids, 
                    attention_mask=attention_mask, 
                    labels=labels
                )
                
                loss = outputs.loss
                loss.backward()
                optimizer.step()
                optimizer.zero_grad()
                
                loop.set_postfix(loss=loss.item())
        
        # Save model
        os.makedirs(output_dir, exist_ok=True)
        self.model.save_pretrained(output_dir)
        self.tokenizer.save_pretrained(output_dir)
        print(f"Model saved to {output_dir}")

    def generate(self, role: str, mandatory_skills: str, optional_skills: str = "", experience_years: int = None, max_length=300) -> str:
        """
        Generates a JD based on structured constraints.
        """
        self.model.eval()
        # Construct structured prompt
        prompt = f"generate job description: role: {role} mandatory: {mandatory_skills}"
        if optional_skills:
            prompt += f" optional: {optional_skills}"
        if experience_years:
            prompt += f" experience: {experience_years} years"
            
        inputs = self.tokenizer(prompt, return_tensors="pt").to(self.device)
        
        with torch.no_grad():
            outputs = self.model.generate(
                inputs.input_ids, 
                max_length=max_length, 
                num_beams=4, 
                early_stopping=True,
                no_repeat_ngram_size=2
            )
        
        return self.tokenizer.decode(outputs[0], skip_special_tokens=True)

