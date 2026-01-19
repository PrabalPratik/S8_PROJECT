from models.t5_generator import T5JDGenerator

if __name__ == "__main__":
    role = "Software Engineer"
    skills = "Python, SQL, AWS"
    generator = T5JDGenerator()
    jd = generator.generate(role, skills)
    print("Generated JD:\n", jd)
