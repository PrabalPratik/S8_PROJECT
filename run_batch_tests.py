
import subprocess
import json

test_cases = [
    {"role": "Backend Developer", "skills": "Java, Spring Boot, Microservices"},
    {"role": "Frontend Engineer", "skills": "React, TypeScript, CSS"},
    {"role": "Cybersecurity Analyst", "skills": "Linux, Ethical Hacking, Network Security"}
]

def run_test(role, skills):
    print(f"\n--- Running Test: {role} ---")
    print(f"Inputs: Role='{role}', Skills='{skills}'")
    
    # Run main.py as a subprocess to capture output
    # We pass role and skills as arguments
    try:
        result = subprocess.run(
            ["python", "main.py", role, skills],
            capture_output=True,
            text=True,
            check=True
        )
        output = result.stdout
        
        # Extract key parts from output (simple parsing for summary)
        adherence = "N/A"
        top_candidate = "N/A"
        top_terms = []
        
        for line in output.split('\n'):
            if "Skill Adherence:" in line:
                adherence = line.split("Skill Adherence:")[1].strip()
            if "1. " in line and "Match Score:" in line:
                top_candidate = line.strip()
            if line.startswith("- ") and ":" in line and any(t in line for t in ["0.", "1."]):
                top_terms.append(line.strip())
                
        print(f"Result: {adherence} Adherence")
        print(f"Top Match: {top_candidate}")
        print(f"Reasoning (Top 3): {', '.join(top_terms[:3])}")
        
    except subprocess.CalledProcessError as e:
        print(f"Error running test: {e}")
        print(e.stderr)

if __name__ == "__main__":
    for tc in test_cases:
        run_test(tc["role"], tc["skills"])
