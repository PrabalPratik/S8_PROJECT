
import pandas as pd
import torch
from torch.utils.data import Dataset
import re

class JDDataset(Dataset):
    def __init__(self, data_path, tokenizer, max_length=512, mode="generation"):
        """
        Args:
            data_path: Path to CSV file.
            tokenizer: HF Tokenizer (T5Tokenizer or BertTokenizer).
            max_length: Max sequence length.
            mode: 'generation' (for T5) or 'ranking' (for BERT).
        """
        self.data = pd.read_csv(data_path)
        
        # Normalize column names to lower_case with underscores
        self.data.columns = [c.lower().replace(' ', '_') for c in self.data.columns]
        
        # Map known columns to standard internal names
        # Standard: job_title, job_description, job_skill_set
        if 'job_title' not in self.data.columns and 'title' in self.data.columns:
            self.data.rename(columns={'title': 'job_title'}, inplace=True)
            
        if 'job_description' not in self.data.columns and 'description' in self.data.columns:
            self.data.rename(columns={'description': 'job_description'}, inplace=True)
            
        # Ensure critical columns exist
        if 'job_title' not in self.data.columns or 'job_description' not in self.data.columns:
            raise ValueError(f"Dataset must contain 'job_title' and 'job_description'. Found: {self.data.columns}")

        # Handle missing skills column
        if 'job_skill_set' not in self.data.columns:
            if 'skills' in self.data.columns:
                self.data.rename(columns={'skills': 'job_skill_set'}, inplace=True)
            else:
                self.data['job_skill_set'] = None # Will extract in __getitem__

        # Drop rows with missing title/description
        self.data = self.data.dropna(subset=['job_title', 'job_description'])
        
        self.tokenizer = tokenizer
        self.max_length = max_length
        self.mode = mode

    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, idx):
        row = self.data.iloc[idx]
        
        title = self.clean_text(row['job_title'])
        description = self.clean_text(row['job_description'])
        
        if row['job_skill_set'] is None or pd.isna(row['job_skill_set']):
            skills = self.extract_skills(description)
        else:
            skills = self.clean_text(row['job_skill_set'])
        
        if self.mode == "generation":
            # T5 Input Format: "generate job description: role: [ROLE] skills: [SKILLS]"
            input_text = f"generate job description: role: {title} skills: {skills}"
            target_text = description
            
            # Tokenize
            model_inputs = self.tokenizer(
                input_text, 
                max_length=self.max_length, 
                padding="max_length", 
                truncation=True, 
                return_tensors="pt"
            )
            
            with self.tokenizer.as_target_tokenizer():
                labels = self.tokenizer(
                    target_text, 
                    max_length=self.max_length, 
                    padding="max_length", 
                    truncation=True, 
                    return_tensors="pt"
                )
            
            return {
                "input_ids": model_inputs.input_ids.squeeze(),
                "attention_mask": model_inputs.attention_mask.squeeze(),
                "labels": labels.input_ids.squeeze()
            }
            
        elif self.mode == "ranking":
            # BERT Input: Resume (simulated here) + JD
            # For this demo, we mock a resume or just return raw text for the ranker wrapper
            pass

    def clean_text(self, text):
        if not isinstance(text, str):
            return ""
        text = re.sub(r'\s+', ' ', text)  # remove extra whitespace
        return text.strip()

    def extract_skills(self, description):
        """
        Simple heuristic to extract skills from description if missing.
        Looks for common headers like "Skills", "Requirements", etc.
        """
        if not description:
            return "General"
            
        # Try to find a 'Skills' or 'Requirements' section
        match = re.search(r'(skills|requirements|qualifications|proficient in|experience with)[:\n]', description, re.IGNORECASE)
        if match:
            # Take the next 300 characters as likely skills
            start = match.end()
            return description[start:start+300].strip()
        
        # Fallback: Keywords (very naive list, can be expanded)
        keywords = ["python", "java", "sql", "excel", "communication", "management", "analysis", "aws", "azure", "react", "node", "c++", "leadership", "agile"]
        found = [k for k in keywords if k in description.lower()]
        if found:
            return ", ".join(found)
            
        return "General"

