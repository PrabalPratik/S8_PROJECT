import re

def calculate_skill_adherence(generated_jd, required_skills):
    """
    Calculates the percentage of required skills present in the generated JD.
    required_skills: string or list of skills
    """
    if not generated_jd or not required_skills:
        return 0.0
        
    if isinstance(required_skills, str):
        # Split by common delimiters
        skills_list = [s.strip().lower() for s in re.split(r'[,|;]', required_skills) if s.strip()]
    else:
        skills_list = [s.lower() for s in required_skills]
        
    if not skills_list:
        return 1.0
        
    jd_lower = generated_jd.lower()
    found_count = 0
    for skill in skills_list:
        if skill in jd_lower:
            found_count += 1
            
    return found_count / len(skills_list)

def calculate_cvm(generated_jd, mandatory_skills):
    """
    Calculates Constraint Violation Metric (CVM).
    CVM = fraction of MANDATORY skills MISSING from the JD.
    Lower is better (0.0 = perfect adherence).
    """
    if not generated_jd or not mandatory_skills:
        # If no mandatory skills, 0 violation. If minimal JD but constraints exist, 100% violation?
        # Logic: If mandatory skills provided, we check them.
        if mandatory_skills: return 1.0 # JD empty but skills required
        return 0.0

    if isinstance(mandatory_skills, str):
         skills_list = [s.strip().lower() for s in re.split(r'[,|;]', mandatory_skills) if s.strip()]
    else:
        skills_list = [s.lower() for s in mandatory_skills]
    
    if not skills_list:
        return 0.0
        
    jd_lower = generated_jd.lower()
    missing_count = 0
    for skill in skills_list:
        if skill not in jd_lower:
            missing_count += 1
            
    return missing_count / len(skills_list)
