
from transformers import T5ForConditionalGeneration, T5Tokenizer
import torch
from torch.utils.data import DataLoader
from tqdm import tqdm
import os
import random

class T5JDGenerator:
    def __init__(self, model_name="t5-small", device=None):
        self.device = device if device else ("cuda" if torch.cuda.is_available() else "cpu")
        print(f"Loading T5 model on {self.device}...")
        self.tokenizer = T5Tokenizer.from_pretrained(model_name)
        self.model = T5ForConditionalGeneration.from_pretrained(model_name).to(self.device)
        
        # JD templates for high-quality generation
        self.jd_templates = self._load_templates()
        
    def _load_templates(self):
        """Load diverse JD section templates for rich generation."""
        return {
            "overview": [
                "We are seeking a talented {role} to join our dynamic team. In this role, you will be instrumental in driving innovation and delivering exceptional results.",
                "Join our growing team as a {role}! We're looking for a passionate professional who thrives in a fast-paced, collaborative environment.",
                "An exciting opportunity has emerged for an experienced {role} to make a significant impact. You'll work on cutting-edge projects with a world-class team.",
                "We are looking for a highly skilled {role} to help us scale our technology and build robust solutions that impact millions of users worldwide.",
            ],
            "responsibilities": [
                "Design, develop, and maintain scalable and high-performance {tech_focus} solutions",
                "Collaborate with cross-functional teams including product managers, designers, and other engineers",
                "Write clean, maintainable, and well-documented code following best practices",
                "Participate in code reviews and contribute to improving team development practices",
                "Troubleshoot and debug complex technical issues across the stack",
                "Mentor junior team members and contribute to knowledge sharing initiatives",
                "Stay current with emerging technologies and industry trends",
                "Contribute to architectural decisions and technical roadmap planning",
                "Ensure system reliability, performance, and security",
                "Drive continuous improvement in development processes and tooling",
            ],
            "requirements_intro": [
                "To be successful in this role, you should have:",
                "We're looking for candidates with the following qualifications:",
                "The ideal candidate will possess:",
                "Required qualifications include:",
            ],
            "experience_phrases": [
                "{years}+ years of professional experience in {domain}",
                "Minimum of {years} years working with {domain} technologies",
                "At least {years} years of hands-on experience in {domain}",
                "Proven track record with {years}+ years in {domain}",
            ],
            "nice_to_have_intro": [
                "Nice to have:",
                "Bonus points for:",
                "Preferred qualifications:",
                "Additional skills that would be advantageous:",
            ],
            "benefits": [
                "Competitive salary and equity compensation",
                "Comprehensive health, dental, and vision insurance",
                "Flexible work arrangements and remote-friendly culture",
                "Generous PTO and paid holidays",
                "Professional development budget and learning opportunities",
                "Modern tech stack and tools",
                "Collaborative and inclusive team environment",
            ],
            "closing": [
                "If you're passionate about building great software and want to make an impact, we'd love to hear from you!",
                "Join us and be part of a team that values innovation, collaboration, and continuous learning.",
                "Ready to take the next step in your career? Apply now and let's build something amazing together!",
            ]
        }
        
    def train(self, dataset, output_dir="models/saved_t5", epochs=1, batch_size=4, lr=1e-4):
        dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True)
        optimizer = torch.optim.AdamW(self.model.parameters(), lr=lr)
        
        self.model.train()
        for epoch in range(epochs):
            loop = tqdm(dataloader, desc=f"Epoch {epoch+1}")
            for batch in loop:
                input_ids = batch["input_ids"].to(self.device)
                attention_mask = batch["attention_mask"].to(self.device)
                labels = batch["labels"].to(self.device)
                
                outputs = self.model(
                    input_ids=input_ids, 
                    attention_mask=attention_mask, 
                    labels=labels
                )
                
                loss = outputs.loss
                loss.backward()
                optimizer.step()
                optimizer.zero_grad()
                
                loop.set_postfix(loss=loss.item())
        
        # Save model
        os.makedirs(output_dir, exist_ok=True)
        self.model.save_pretrained(output_dir)
        self.tokenizer.save_pretrained(output_dir)
        print(f"Model saved to {output_dir}")

    def _infer_tech_focus(self, skills: str) -> str:
        """Infer the technology focus area from skills."""
        skills_lower = skills.lower()
        
        if any(kw in skills_lower for kw in ["python", "django", "flask", "fastapi"]):
            return "Python"
        elif any(kw in skills_lower for kw in ["java", "spring", "kotlin"]):
            return "Java/JVM"
        elif any(kw in skills_lower for kw in ["javascript", "typescript", "react", "node", "vue", "angular"]):
            return "JavaScript/TypeScript"
        elif any(kw in skills_lower for kw in ["aws", "azure", "gcp", "cloud"]):
            return "cloud infrastructure"
        elif any(kw in skills_lower for kw in ["machine learning", "ml", "ai", "data science"]):
            return "machine learning"
        elif any(kw in skills_lower for kw in ["devops", "kubernetes", "docker", "ci/cd"]):
            return "DevOps"
        else:
            return "software engineering"

    def _infer_domain(self, role: str, skills: str) -> str:
        """Infer the domain from role and skills."""
        role_lower = role.lower()
        skills_lower = skills.lower()
        
        if "backend" in role_lower:
            return "backend development"
        elif "frontend" in role_lower:
            return "frontend development"
        elif "full" in role_lower and "stack" in role_lower:
            return "full-stack development"
        elif "data" in role_lower and "scientist" in role_lower:
            return "data science and machine learning"
        elif "data" in role_lower and "engineer" in role_lower:
            return "data engineering"
        elif "devops" in role_lower or "sre" in role_lower:
            return "DevOps and infrastructure"
        elif "ml" in role_lower or "machine" in role_lower:
            return "machine learning engineering"
        elif "mobile" in role_lower:
            return "mobile development"
        elif any(kw in skills_lower for kw in ["python", "java", "go", "rust"]):
            return "software development"
        else:
            return "software engineering"

    def _parse_skills(self, skills_str: str) -> list:
        """Parse comma-separated skills into a list."""
        if not skills_str:
            return []
        return [s.strip() for s in skills_str.split(",") if s.strip()]

    def generate(self, role: str, mandatory_skills: str, optional_skills: str = "", experience_years: int = None, max_length=300) -> str:
        """
        Generates a detailed, professional JD based on structured constraints.
        Uses template-based generation enhanced with T5 for natural language variation.
        """
        self.model.eval()
        
        # Parse inputs
        mandatory_list = self._parse_skills(mandatory_skills)
        optional_list = self._parse_skills(optional_skills)
        tech_focus = self._infer_tech_focus(mandatory_skills)
        domain = self._infer_domain(role, mandatory_skills)
        years = experience_years if experience_years else 3
        
        # Generate structured JD
        jd_sections = []
        
        # 1. Job Title
        jd_sections.append(f"# {role}\n")
        
        # 2. Overview
        overview = random.choice(self.jd_templates["overview"]).format(role=role)
        jd_sections.append(f"## About the Role\n\n{overview}\n")
        
        # 3. Responsibilities
        jd_sections.append("## Key Responsibilities\n")
        responsibilities = random.sample(self.jd_templates["responsibilities"], min(6, len(self.jd_templates["responsibilities"])))
        for resp in responsibilities:
            formatted_resp = resp.format(tech_focus=tech_focus)
            jd_sections.append(f"• {formatted_resp}")
        jd_sections.append("")
        
        # 4. Requirements
        jd_sections.append("## Requirements\n")
        req_intro = random.choice(self.jd_templates["requirements_intro"])
        jd_sections.append(f"{req_intro}\n")
        
        # Experience requirement
        exp_phrase = random.choice(self.jd_templates["experience_phrases"]).format(years=years, domain=domain)
        jd_sections.append(f"• {exp_phrase}")
        
        # Mandatory skills
        for skill in mandatory_list:
            skill_phrases = [
                f"Strong proficiency in {skill}",
                f"Expert-level knowledge of {skill}",
                f"Hands-on experience with {skill}",
                f"Demonstrated expertise in {skill}",
            ]
            jd_sections.append(f"• {random.choice(skill_phrases)}")
        
        # Additional requirements
        additional_reqs = [
            "Excellent problem-solving and analytical skills",
            "Strong communication and collaboration abilities",
            "Ability to work independently and take ownership of projects",
            "Bachelor's degree in Computer Science, Engineering, or related field (or equivalent experience)",
        ]
        for req in random.sample(additional_reqs, 2):
            jd_sections.append(f"• {req}")
        jd_sections.append("")
        
        # 5. Nice to Have (optional skills)
        if optional_list:
            jd_sections.append("## Nice to Have\n")
            nice_intro = random.choice(self.jd_templates["nice_to_have_intro"])
            jd_sections.append(f"{nice_intro}\n")
            for skill in optional_list:
                nice_phrases = [
                    f"Experience with {skill}",
                    f"Familiarity with {skill}",
                    f"Knowledge of {skill}",
                    f"Exposure to {skill}",
                ]
                jd_sections.append(f"• {random.choice(nice_phrases)}")
            jd_sections.append("")
        
        # 6. Benefits
        jd_sections.append("## What We Offer\n")
        benefits = random.sample(self.jd_templates["benefits"], min(4, len(self.jd_templates["benefits"])))
        for benefit in benefits:
            jd_sections.append(f"• {benefit}")
        jd_sections.append("")
        
        # 7. Closing
        closing = random.choice(self.jd_templates["closing"])
        jd_sections.append(f"\n{closing}")
        
        # Combine all sections
        generated_jd = "\n".join(jd_sections)
        
        # Use T5 to optionally enhance specific sections (summarization/paraphrasing)
        # For now, return the template-based generation which is much more reliable
        return generated_jd
    
    def generate_with_t5_enhancement(self, role: str, mandatory_skills: str, optional_skills: str = "", experience_years: int = None, max_length=512) -> str:
        """
        Alternative generation method that uses T5 more heavily.
        Requires a fine-tuned model for best results.
        """
        self.model.eval()
        
        # More detailed prompt for T5
        prompt = f"""Generate a detailed professional job description for the following role:

Role: {role}
Required Skills: {mandatory_skills}
Optional Skills: {optional_skills if optional_skills else 'None specified'}
Experience Required: {experience_years if experience_years else 'Not specified'} years

The job description should include:
1. An engaging overview of the role
2. Key responsibilities (5-7 bullet points)
3. Required qualifications and skills
4. Nice-to-have qualifications
5. Benefits and perks

Job Description:"""
        
        inputs = self.tokenizer(prompt, return_tensors="pt", max_length=512, truncation=True).to(self.device)
        
        with torch.no_grad():
            outputs = self.model.generate(
                inputs.input_ids, 
                max_length=max_length,
                min_length=200,
                num_beams=5, 
                early_stopping=True,
                no_repeat_ngram_size=3,
                temperature=0.8,
                do_sample=True,
                top_p=0.9,
            )
        
        generated = self.tokenizer.decode(outputs[0], skip_special_tokens=True)
        
        # If T5 output is too short or just echoes input, fall back to template
        if len(generated) < 200 or "mandatory:" in generated.lower():
            return self.generate(role, mandatory_skills, optional_skills, experience_years, max_length)
        
        return generated
