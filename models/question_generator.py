from typing import List, Dict, Any, Optional
import random

class QuestionGenerator:
    """
    Generates dynamic interview questions based on candidate profile.
    Focuses on Projects and Experience.
    """
    
    def __init__(self):
        # Rule-based templates effectively cover common scenarios without LLM overhead
        self.project_templates = [
            "You mentioned working on '{project}'. Can you describe the most significant technical challenge you faced?",
            "In your '{project}', what specific role did you play in the implementation of {tech}?",
            "How did you ensure scalability and performance in the '{project}'?",
            "What alternative technologies did you consider for '{project}' and why did you choose {tech}?"
        ]
        
        self.experience_templates = [
            "With {years} years of experience, describe a situation where you had to mentor a junior developer.",
            "In your {years} years, what is the most complex system architecture you designed or maintained?",
            "Given your experience, how do you approach debugging a critical production issue?",
            "Can you discuss a time when you had to advocate for a technical decision against resistance?"
        ]
        
        self.skill_templates = [
            "How do you stay updated with the latest changes in {skill}?",
            "Can you explain a complex concept in {skill} to a non-technical stakeholder?",
            "What are the common pitfalls you've encountered when using {skill} in production?"
        ]

    def generate_questions(self, candidate_data: Dict[str, Any], num_questions: int = 3) -> List[str]:
        """
        Generates a tailored list of interview questions.
        
        candidate_data expected structure:
        - projects: list of {"name": str, "tech": list of str}
        - experience_years: int
        - skills: list of str
        """
        questions = []
        
        # 1. Project-Based Questions
        projects = candidate_data.get("projects", [])
        if projects:
            # Pick a random project to focus on
            proj = random.choice(projects)
            tech = random.choice(proj.get("tech", ["relevant technologies"])) if proj.get("tech") else "the core stack"
            
            q_template = random.choice(self.project_templates)
            q = q_template.format(project=proj.get("name", "your recent project"), tech=tech)
            questions.append(q)
            
        # 2. Experience-Based Questions
        years = candidate_data.get("experience_years", 0)
        if years > 2:
            q_template = random.choice(self.experience_templates)
            q = q_template.format(years=years)
            questions.append(q)
            
        # 3. Skill-Based Questions (Fill remaining slots)
        skills = candidate_data.get("skills", [])
        while len(questions) < num_questions and skills:
            skill = random.choice(skills)
            q_template = random.choice(self.skill_templates)
            q = q_template.format(skill=skill)
            if q not in questions:
                questions.append(q)
                
        # Fallback if we still don't have enough questions
        if len(questions) < num_questions:
            questions.append("What motivates you to apply for this specific role?")
            
        return questions[:num_questions]
