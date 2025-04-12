import json
import random
from faker import Faker
from datetime import datetime, timedelta
from pathlib import Path

fake = Faker()

# Seed for reproducibility
Faker.seed(42)
random.seed(42)


# Helper function to generate a fake course
def generate_course(course_id, title, subject):
    grade_choices = ["A", "B", "C", "D", "F"]
    grade = random.choices(grade_choices, weights=[0.35, 0.3, 0.2, 0.1, 0.05])[0]
    grade_value = {"A": 40.0, "B": 30.0, "C": 20.0, "D": 10.0, "F": 0.0}[grade]
    return {
        "course_id": course_id,
        "course_title": title,
        "credits_attempted": 3.0,
        "credits_earned": 3.0 if grade != "F" else 0.0,
        "final_grade": grade,
        "grade_value": grade_value,
        "subject": subject,
        "is_transferred": "N",
        "is_withdrawn": "N",
        "course_level": "Undergraduate",
        "instruction_method": "Face-to-Face Instruction",
        "course_passed_ind": "Y" if grade != "F" else "N",
        "course_failed_ind": "N" if grade != "F" else "Y",
    }


# Generate academic advisor-style queries
def generate_academic_data_entry(index):
    name = fake.name()
    first_name, last_name = name.split()[0], name.split()[-1]
    full_name = f"{first_name} {last_name}"
    student_id = random.randint(10000000, 99999999)
    gpa = round(random.uniform(2.0, 4.0), 2)
    cgpa = round(min(4.0, gpa + random.uniform(-0.2, 0.2)), 2)
    semester = fake.date_between(start_date="-10y", end_date="today")
    semester_desc = f"{semester.strftime('%B')} {semester.year}"
    academic_standing = "Good Standing" if gpa >= 2.5 else "Academic Probation"
    academic_period = int(f"{semester.year}{random.randint(10, 90)}")

    courses = [
        generate_course("MATH1241", "Calculus I", "Mathematics"),
        generate_course("ITCS1212", "Intro to Computer Science", "Computer Science"),
        generate_course("PHYS1101", "Physics I", "Physics"),
    ]

    query = f"Can you provide a performance summary for {full_name} in {random.choice([c['subject'] for c in courses])} during {semester_desc}?"

    student_info = {
        "demographic": {
            "student_id": student_id,
            "name": name,
            "first_name": first_name,
            "last_name": last_name,
            "current_age": random.randint(18, 40),
            "gender": random.choice(["Male", "Female", "Other"]),
            "citizenship": "US",
            "nationality": fake.country(),
            "marital_status": random.choice(["Single", "Married", "Divorced"]),
            "employment_type": None,
            "native_language": None,
            "spouse_accompanied_ind": None,
            "children_number": None,
            "veteran_category": random.choice(["Veteran", "None"]),
            "primary_ethnicity": random.choice(
                ["Asian", "White", "Black", "Latino", "Other"]
            ),
        },
        "semester": {
            "academic_period": academic_period,
            "academic_period_desc": semester_desc,
            "student_population": "Undergraduate",
            "semester_time_status": random.choice(["Full Time", "Half Time"]),
            "addmission_population": "Readmit",
            "enrollment_status": "Eligible to Register",
            "academic_standing": academic_standing,
            "student_status": "Active",
            "fin_aid_applicant_ind": random.choice(["Y", "N"]),
            "housing_ind": random.choice(["Y", "N"]),
            "gpa": gpa,
            "cgpa": cgpa,
            "courses": courses,
        },
    }

    subjects = ", ".join([c["course_title"] for c in courses])
    performance = (
        f"{first_name} earned grades of "
        + ", ".join([f"{c['final_grade']} in {c['course_title']}" for c in courses])
        + "."
    )
    summary = f"<think>{first_name} {last_name} had a GPA of {gpa} and a cumulative GPA of {cgpa} during {semester_desc}, with an academic standing of '{academic_standing}'. They completed the following courses: {subjects}. {performance}</think>"

    return {
        "text": f"<|im_start|>user\n{query}\n\nRelevant Information:\n- {json.dumps(student_info)}<|im_end|>\n<|im_start|>assistant\n{summary}<|im_end|>"
    }


# Generate 100 examples
examples = [generate_academic_data_entry(i) for i in range(200)]

# Save to JSONL
output_path = Path("coldstart_data/academic_advisor_coldstart.jsonl")
output_path.parent.mkdir(parents=True, exist_ok=True)

with output_path.open("w", encoding="utf-8") as f:
    for example in examples:
        f.write(json.dumps(example) + "\n")

print(f"Coldstart examples written to {output_path.resolve()}")
