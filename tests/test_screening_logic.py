
import sys
import os

# Add parent directory to path to import modules
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from utils.screening import ScreeningEngine
from models.question_generator import QuestionGenerator

def test_screening_engine():
    print("Testing ScreeningEngine...")
    engine = ScreeningEngine()
    
    # Test Banding
    # Prime: > 85% and 0 missing
    cat1 = engine.categorize_candidate(0.89, 0)
    assert cat1["band"] == "Prime", f"Expected Prime, got {cat1['band']}"
    
    # Contender: 88% but missing 1 skill OR 75%
    cat2 = engine.categorize_candidate(0.88, 1)
    assert cat2["band"] == "Contender", f"Expected Contender, got {cat2['band']}"
    
    cat3 = engine.categorize_candidate(0.75, 0)
    assert cat3["band"] == "Contender", f"Expected Contender, got {cat3['band']}"
    
    print("✅ ScreeningEngine Banding Logic Passed")

    # Test Feedback
    feedback = engine.generate_feedback(
        {"semantic_score": 0.88, "missing_skills": ["Java"], "exp_years": 4},
        {"mandatory_skills": ["Python", "Java"], "min_exp": 5}
    )
    print(f"Sample Feedback: {feedback}")
    assert "Java" in feedback
    assert "gap" in feedback
    print("✅ ScreeningEngine Feedback Logic Passed")

def test_question_generator():
    print("\nTesting QuestionGenerator...")
    gen = QuestionGenerator()
    
    cand_data = {
        "projects": [{"name": "E-commerce Bot", "tech": ["Python"]}],
        "experience_years": 5,
        "skills": ["AWS", "Docker"]
    }
    
    questions = gen.generate_questions(cand_data)
    for i, q in enumerate(questions, 1):
        print(f"Q{i}: {q}")
        
    assert len(questions) == 3
    assert "E-commerce Bot" in questions[0]
    print("✅ QuestionGenerator Logic Passed")

if __name__ == "__main__":
    test_screening_engine()
    test_question_generator()
