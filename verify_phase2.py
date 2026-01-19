from models.t5_generator import T5JDGenerator
from utils.metrics import calculate_cvm

def verify_phase2():
    print("=== Phase 2 Verification: Structured T5 & CVM ===\n")
    
    # 1. Test Structured Generation
    print("[1/2] Structured Generation...")
    t5 = T5JDGenerator()
    role = "Backend Engineer"
    mandatory = "Python, Django, PostgreSQL" # These MUST be in JD
    optional = "AWS, Docker" # Nice to have
    
    print(f"Input Role: {role}")
    print(f"Mandatory: {mandatory}")
    print(f"Optional: {optional}")
    
    jd = t5.generate(role, mandatory, optional, experience_years=3)
    print(f"\nGenerated JD Preview:\n{jd[:200]}...\n")
    
    # 2. Test CVM (Constraint Violation Metric)
    print("[2/2] Testing CVM Metric...")
    
    # Simulation A: Perfect JD
    jd_perfect = "We need a Backend Engineer with strong Python and Django skills. PostgreSQL experience is required."
    cvm_perfect = calculate_cvm(jd_perfect, mandatory)
    print(f"Perfect JD CVM (Expected 0.0): {cvm_perfect:.2f}")
    
    # Simulation B: Missing PostgreSQL
    jd_flawed = "We need a Backend Engineer with strong Python and Django skills."
    cvm_flawed = calculate_cvm(jd_flawed, mandatory)
    print(f"Flawed JD CVM (Expected ~0.33): {cvm_flawed:.2f}") # 1 out of 3 missing

    if cvm_perfect == 0.0 and cvm_flawed > 0.0:
        print("\nSUCCESS: CVM logic verified.")
    else:
        print("\nFAILURE: CVM logic mismatch.")

if __name__ == "__main__":
    verify_phase2()
