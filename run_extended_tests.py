import sys
import re
from models.t5_generator import T5JDGenerator
from models.bert_ranker import BertRanker
from utils.metrics import calculate_skill_adherence

# Define 10 diverse test cases (role, skills)
TEST_CASES = [
    ("Data Scientist", "Python, SQL, Machine Learning"),
    ("Backend Developer", "Java, Spring Boot, Microservices"),
    ("Frontend Engineer", "JavaScript, React, CSS"),
    ("DevOps Engineer", "Docker, Kubernetes, CI/CD"),
    ("Cybersecurity Analyst", "Network Security, Ethical Hacking, Penetration Testing"),
    ("Product Manager", "Agile, Roadmapping, Stakeholder Management"),
    ("Research Scientist", "Deep Learning, PyTorch, TensorFlow"),
    ("Mobile Developer", "Flutter, Dart, iOS, Android"),
    ("QA Engineer", "Test Automation, Selenium, pytest"),
    ("Business Analyst", "SQL, Excel, Data Visualization")
]


def run_test(role, skills):
    # Initialize models (small models for quick demo)
    t5 = T5JDGenerator(model_name="t5-small")
    jd = t5.generate(role, skills)
    adherence = calculate_skill_adherence(jd, skills)
    # Load BERT ranker
    ranker = BertRanker(model_name="bert-base-uncased")
    # Use a single dummy resume (first row from resumes.csv) for demonstration
    import pandas as pd
    resumes = pd.read_csv("data/resumes.csv")
    first_resume = resumes.iloc[0]
    resume_text = f"Skills: {first_resume['Skills']}. Experience: {first_resume['Experience (Years)']} years. Education: {first_resume['Education']}."
    score = ranker.get_hybrid_score(resume_text, jd, required_skills=skills)
    return jd, adherence, score


def main():
    print("=== Running Extended Test Suite (10 cases) ===")
    for idx, (role, skills) in enumerate(TEST_CASES, 1):
        try:
            jd, adherence, score = run_test(role, skills)
            print(f"\nTest {idx}: Role='{role}', Skills='{skills}'")
            print(f"Generated JD (first 120 chars): {jd[:120]}...")
            print(f"Skill Adherence: {adherence:.2%}")
            print(f"Hybrid Rank Score (dummy resume): {score:.4f}")
        except Exception as e:
            print(f"Error in test {idx}: {e}")
            # Continue to next test case

if __name__ == "__main__":
    main()
