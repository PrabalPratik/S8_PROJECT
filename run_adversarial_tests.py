from models.bert_ranker import BertRanker
import numpy as np

def run_adversarial_tests():
    print("=== Phase 4: Adversarial & Robustness Stress Test ===\n")
    
    ranker = BertRanker()
    jd_text = "Software Engineer. Python, SQL."
    mandatory = "Python, SQL"
    
    # 1. Keyword Stuffing
    # Candidate repeats "Python" 50 times.
    stuffed_resume = "Python " * 50
    normal_resume = "I am a Software Engineer with Python and SQL skills."
    
    print("[Test 1] Keyword Stuffing vs Normal Resume")
    score_stuffed, _ = ranker.get_research_score(stuffed_resume, jd_text, mandatory)
    score_normal, _ = ranker.get_research_score(normal_resume, jd_text, mandatory)
    
    print(f"Stuffed Score: {score_stuffed:.4f}")
    print(f"Normal Score:  {score_normal:.4f}")
    
    # We expect normal to be higher (BERT contextual embeddings dislike varied repitition vs coherent semantic sentences)
    # Although raw bag-of-words might love it, BERT is contextual. Only CLS token is used.
    if score_normal >= score_stuffed:
        print(">> PASS: Contextual semantics preferred over stuffing.")
    else:
        print(">> FAIL: Model fooled by keyword stuffing.")

    # 2. Empty/Gibberish Input
    print("\n[Test 2] Empty or Gibberish Input")
    try:
        score_empty, _ = ranker.get_research_score("", jd_text, mandatory)
        print(f"Empty Input Score: {score_empty:.4f}")
        
        score_gibberish, _ = ranker.get_research_score("adjfklasjdflaskjf", jd_text, mandatory)
        print(f"Gibberish Score: {score_gibberish:.4f}")
        
        if score_empty < 0.1 and score_gibberish < 0.2:
             print(">> PASS: Low scores for invalid inputs.")
        else:
             print(">> FAIL: High scores for garbage.")
             
    except Exception as e:
        print(f">> ERROR: Crash on invalid input - {e}")

    # 3. Missing Mandatory Skills (Explicit)
    print("\n[Test 3] Semantic Match but Missing Constraint")
    # "I love coding" semantically matches "Software Engineer" somewhat, but lacks "Python, SQL"
    semantic_resume = "I am a passionate coder who loves building software systems."
    score_semantic, info = ranker.get_research_score(semantic_resume, jd_text, mandatory)
    
    print(f"Score: {score_semantic:.4f} | Sim: {info['sim']:.4f} | Penalty: {info['pen_miss']:.4f}")
    
    if info['pen_miss'] >= 0.2: # 2 skills missing -> 1.0 * Beta(0.2) = 0.2
        print(">> PASS: Full penalty applied for missing constraints.")
    else:
        print(f">> FAIL: Penalty too low ({info['pen_miss']}). skills detection failed?")

if __name__ == "__main__":
    run_adversarial_tests()
