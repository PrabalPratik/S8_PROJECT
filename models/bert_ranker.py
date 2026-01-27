
import re
import torch
import torch.nn as nn
from transformers import AutoModel, AutoTokenizer

class BertRanker(nn.Module):
    def __init__(self, model_name="bert-base-uncased", device=None):
        super(BertRanker, self).__init__()
        self.device = device if device else ("cuda" if torch.cuda.is_available() else "cpu")
        print(f"Loading BERT Ranker model ({model_name}) on {self.device}...")
        
        self.bert = AutoModel.from_pretrained(model_name).to(self.device)
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        
        # Simple scorer: Cosine similarity or a regression head
        # Here we'll use a regression head on top of the [CLS] token of the concatenated input
        # OR Siamese style (separate embeddings). 
        # Plan: Cross-Encoder style (Resume [SEP] JD) is usually more accurate for ranking than Siamese.
        # Let's use Cross-Encoder: Input = "[CLS] Resume [SEP] JD [SEP]" -> Output = Score
        self.score_head = nn.Linear(self.bert.config.hidden_size, 1).to(self.device)
        self.sigmoid = nn.Sigmoid()
        
    def get_research_score(self, resume_text, jd_text, mandatory_skills=None, experience_gap=0, alpha=0.7, beta=0.2, gamma=0.1, skill_rarity_weights=None):
        """
        Formalized Hybrid Scoring Equation:
        Score = alpha * Sim(Resume, JD) - beta * Penalty(Missing) - gamma * Penalty(Exp)
        
        Enhanced with Skill Rarity Weighting (SRW):
        When skill_rarity_weights is provided, missing rare skills are penalized more heavily.
        
        Args:
            resume_text: Candidate resume text
            jd_text: Job description text
            mandatory_skills: Required skills (string or list)
            experience_gap: Years of experience shortfall
            alpha: Semantic similarity weight
            beta: Skill penalty weight
            gamma: Experience gap weight
            skill_rarity_weights: Optional dict mapping skill -> rarity weight (0-1)
        """
        # 1. Semantic Similarity (Sim)
        sim_score = self.predict(resume_text, jd_text)
        
        # 2. Constraint Penalty (Beta) with optional SRW
        penalty_missing = 0.0
        missing_skills = []
        skills_list = []
        
        if mandatory_skills:
            if isinstance(mandatory_skills, str):
                skills_list = [s.strip().lower() for s in re.split(r'[,|;]', mandatory_skills) if s.strip()]
            else:
                skills_list = [s.lower() for s in mandatory_skills]
            
            if skills_list:
                resume_lower = resume_text.lower()
                for s in skills_list:
                    if s not in resume_lower:
                        missing_skills.append(s)
                
                if skill_rarity_weights:
                    # SRW: Apply weighted penalty based on skill rarity
                    total_weight = 0.0
                    for skill in missing_skills:
                        # Get rarity weight (default to 0.5 for unknown skills)
                        weight = skill_rarity_weights.get(skill, 0.5)
                        total_weight += weight
                    
                    # Normalize by max possible weight
                    max_possible = len(skills_list)
                    penalty_missing = total_weight / max_possible if max_possible > 0 else 0
                else:
                    # Standard penalty
                    penalty_missing = len(missing_skills) / len(skills_list)

        # 3. Experience Penalty (Gamma)
        # experience_gap should be > 0 if candidate lacks years. e.g. Req=5, Cand=3 -> Gap=2
        penalty_exp = min(1.0, max(0.0, experience_gap / 10.0)) # Normalize gap (cap at 10 years)
        
        # Final Equation
        final_score = (alpha * sim_score) - (beta * penalty_missing) - (gamma * penalty_exp)
        
        return max(0.0, final_score), {
            "sim": sim_score, 
            "pen_miss": penalty_missing, 
            "pen_exp": penalty_exp,
            "missing_skills": missing_skills,
            "srw_enabled": skill_rarity_weights is not None
        }
    def forward(self, input_ids, attention_mask, token_type_ids=None, output_attentions=False):
        outputs = self.bert(
            input_ids=input_ids, 
            attention_mask=attention_mask, 
            token_type_ids=token_type_ids, 
            output_attentions=output_attentions
        )
        cls_output = outputs.last_hidden_state[:, 0, :] # [CLS] token
        score = self.score_head(cls_output)
        
        if output_attentions:
            return self.sigmoid(score), outputs.attentions
            
        return self.sigmoid(score)

    def predict(self, resume_text, jd_text):
        """
        Predicts suitability score (0-1) for a resume-JD pair.
        """
        self.eval()
        inputs = self.tokenizer(
            resume_text, 
            jd_text, 
            padding=True, 
            truncation=True, 
            max_length=512, 
            return_tensors="pt"
        ).to(self.device)
        
        with torch.no_grad():
            score = self.forward(inputs.input_ids, inputs.attention_mask, inputs.token_type_ids)
        
        return score.item()

    def train_step(self, resume_texts, jd_texts, labels, optimizer, criterion):
        """
        Batch training step.
        resume_texts: list of strings
        jd_texts: list of strings
        labels: Tensor of shape [batch_size]
        """
        self.train()
        inputs = self.tokenizer(
            resume_texts, 
            jd_texts, 
            padding=True, 
            truncation=True, 
            max_length=512, 
            return_tensors="pt"
        ).to(self.device)
        
        # Ensure labels are [Batch, 1] floats
        target = labels.to(self.device).float()
        if len(target.shape) == 1:
            target = target.unsqueeze(1)
        
        optimizer.zero_grad()
        score = self.forward(inputs.input_ids, inputs.attention_mask, inputs.token_type_ids)
        
        loss = criterion(score, target)
        loss.backward()
        optimizer.step()
        
        return loss.item()
