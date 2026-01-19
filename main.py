import pandas as pd
import torch
from models.t5_generator import T5JDGenerator
from models.bert_ranker import BertRanker
from models.attention_utils import explain_prediction, format_explanation
from utils.metrics import calculate_skill_adherence
import sys
import re

def main():
    print("=== BERT-T5 Hybrid Framework ===")
    
    # 1. Inputs
    # If arguments provided, use them, else ask
    if len(sys.argv) > 2:
        role = sys.argv[1]
        skills = sys.argv[2]
    else: 
        role = input("Enter Job Role (e.g. Data Scientist): ") or "Data Scientist"
        skills = input("Enter Skills (e.g. Python, SQL): ") or "Python, SQL"
    
    print(f"\nTarget: {role} | Skills: {skills}")

    # 2. T5 Generation
    print("\n[1/4] Generating Job Description...")
    t5 = T5JDGenerator(model_name="t5-small")
    jd_text = t5.generate(role, skills)
    
    # Calculate Adherence
    adherence = calculate_skill_adherence(jd_text, skills)
    print(f"\nGenerated JD Preview:\n{jd_text[:300]}...")
    print(f"Skill Adherence: {adherence:.2%}\n")
    
    # 3. Candidate Ranking
    print("[2/4] Ranking Candidates from 'data/resumes.csv'...")
    # Use distilbert if trained, or bert-base-uncased generic
    ranker = BertRanker(model_name="bert-base-uncased")
    
    # Load Resumes
    try:
        df = pd.read_csv("data/resumes.csv")
    except FileNotFoundError:
        print("Error: data/resumes.csv not found.")
        return

    print(f"Loaded {len(df)} resumes.")
    
    results = []
    
    # Predict for each
    # For efficiency we could batch, but loop is fine for demo
    for idx, row in df.iterrows():
        # Construct Resume Text
        # Handling missing values casually for demo
        skills_res = str(row.get('Skills', ''))
        exp = str(row.get('Experience (Years)', '0'))
        edu = str(row.get('Education', ''))
        
        res_text = f"Skills: {skills_res}. Experience: {exp} years. Education: {edu}."
        
        # Use hybrid score (BERT + Skill Penalty)
        score = ranker.get_hybrid_score(res_text, jd_text, required_skills=skills)
        results.append({
            "Name": row.get("Name", f"Candidate {idx}"), 
            "Score": score, 
            "Text": res_text,
            "Role": row.get("Job Role", "Unknown")
        })
        
    # Sort
    results.sort(key=lambda x: x["Score"], reverse=True)
    
    # 4. Show Top Results
    print("\n[3/4] Top 5 Candidates:")
    top_candidates = results[:5]
    for i, p in enumerate(top_candidates):
        print(f"{i+1}. {p['Name']} ({p['Role']}) - Match Score: {p['Score']:.4f}")
        
    # 5. Explainability
    print("\n[4/4] Explaining Top Match...")
    if top_candidates:
        top = top_candidates[0]
        score, expl = explain_prediction(ranker, ranker.tokenizer, top['Text'], jd_text)
        print(f"Candidate: {top['Name']}")
        print(format_explanation(expl))
    
    print("\n=== Done ===")

if __name__ == "__main__":
    main()
