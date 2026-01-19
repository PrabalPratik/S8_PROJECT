from models.bert_ranker import BertRanker
from models.attention_utils import compare_candidates
import torch

def verify_phase3():
    print("=== Phase 3 Verification: Ranking Equation & Contrastive Attention ===\n")
    
    ranker = BertRanker()
    jd_text = "Senior Python Developer with AWS and Kubernetes experience."
    mandatory = "Python, AWS, Kubernetes"
    
    # Candidate A: Full match
    resume_A = "Expert Python developer. Certified in AWS and Kubernetes. 5 years exp."
    # Candidate B: Missing Kubernetes
    resume_B = "Python developer with some AWS knowledge. No container experience."
    
    # 1. Test Formal Equation
    print("[1/2] Research Scoring Equation...")
    # Alpha=0.7 (Sim), Beta=0.2 (Missing), Gamma=0.1 (Exp)
    score_A, info_A = ranker.get_research_score(resume_A, jd_text, mandatory, experience_gap=0)
    score_B, info_B = ranker.get_research_score(resume_B, jd_text, mandatory, experience_gap=0)
    
    print(f"Cand A (Ref): Score={score_A:.4f} | Sim={info_A['sim']:.4f} Pen(Miss)={info_A['pen_miss']:.4f}")
    print(f"Cand B (Lag): Score={score_B:.4f} | Sim={info_B['sim']:.4f} Pen(Miss)={info_B['pen_miss']:.4f}")
    
    if score_A > score_B:
        print("Scoring Logic: PASS (Full match > Partial)")
    else:
        print("Scoring Logic: FAIL")

    # 2. Test Contrastive Attention
    print("\n[2/2] Contrastive Attention (Why A > B?)...")
    token_advantages = compare_candidates(ranker, ranker.tokenizer, jd_text, resume_A, resume_B)
    
    print("Top Advantages for A:")
    for t, diff in token_advantages:
        print(f"  Token '{t}': +{diff:.4f}")
        
    if len(token_advantages) > 0:
         print("Contrastive Attention: PASS")
    else:
         print("Contrastive Attention: FAIL")

if __name__ == "__main__":
    verify_phase3()
