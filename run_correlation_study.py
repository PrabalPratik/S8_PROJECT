import pandas as pd
import numpy as np
from scipy.stats import spearmanr
from models.bert_ranker import BertRanker

def run_correlation_study():
    print("=== Phase 4: Human-System Correlation Study ===\n")
    
    ranker = BertRanker()
    
    # Define a Test Case
    jd_text = "Senior Software Engineer. Must have Python, Django, and AWS. 5+ years experience."
    mandatory = "Python, Django, AWS"
    
    # Synthetic "Ground Truth" (Human Ranks)
    # 1: Perfect Match
    # 2: Good Match (Missing 1 skill)
    # 3: Average Match (Missing 2 skills)
    # 4: Poor Match (Wrong Role)
    # 5: Irrelevant (Chef)
    
    dataset = [
        {"id": 1, "text": "Senior Python Developer with 6 years exp. Expert in Django and AWS.", "human_rank": 1},
        {"id": 2, "text": "Python Developer vs 3 years exp. Knows Django but no cloud.", "human_rank": 2},
        {"id": 3, "text": "Java Developer learning Python. No Django or AWS.", "human_rank": 3},
        {"id": 4, "text": "Junior Frontend Dev. React and CSS only.", "human_rank": 4},
        {"id": 5, "text": "Head Chef with 10 years of culinary management.", "human_rank": 5}
    ]
    
    print(f"JD: {jd_text}")
    print(f"Mandatory: {mandatory}\n")
    
    system_scores = []
    human_ranks = []
    
    print(f"{'ID':<4} | {'Human Rank':<12} | {'Sys Score':<10} | {'Resume Snippet'}")
    print("-" * 60)
    
    for item in dataset:
        # Get Research Score (Alpha=0.7, Beta=0.2, Gamma=0.1)
        # For simplicity, we assume experience gap is 0 or manually estimated for valid exp penalty? 
        # Let's keep it simple: just passing text.
        score, _ = ranker.get_research_score(item["text"], jd_text, mandatory_skills=mandatory)
        
        system_scores.append(score)
        human_ranks.append(item["human_rank"])
        
        print(f"{item['id']:<4} | {item['human_rank']:<12} | {score:.4f}     | {item['text'][:30]}...")

    # Calculate Spearman Correlation
    # We expect NEGATIVE correlation because Rank 1 (Low) should have Score 1.0 (High)
    # Or we can rank system scores: High score -> Rank 1.
    
    # Let's convert system scores to ranks (Higher score = Lower rank number)
    # argsort returns indices that sort array. argsort()[::-1] gives indices of high-to-low.
    # rankdata is easier.
    from scipy.stats import rankdata
    # Rank data assigns rank 1 to lowest value. We want rank 1 to highest score.
    # So we rank (-score).
    system_ranks = rankdata([-s for s in system_scores])
    
    print("\n--- Ranks ---")
    print(f"Human Ranks:  {human_ranks}")
    print(f"System Ranks: {system_ranks}")
    
    correlation, p_value = spearmanr(human_ranks, system_ranks)
    print(f"\nSpearman Correlation: {correlation:.4f} (p={p_value:.4f})")
    
    if correlation > 0.8:
        print("Result: STRONG ALIGNMENT with Human Judgment")
    else:
        print("Result: WEAK ALIGNMENT")

if __name__ == "__main__":
    run_correlation_study()
