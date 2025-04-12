import re
import json
from util.logger import ColorLogger

logger = ColorLogger(__name__)
# DEMOGRAPHIC GENERATE SETTINGS
# output_ids = self.model.generate(
#     input_ids,
#     attention_mask=attention_mask,  # Passes attention mask to avoid issues
#     max_new_tokens=1024,
#     temperature=0.55,  # recommended by deep seek = 0.6
#     top_p=0.90,  # default = 1.0
#     top_k=40,  # default = 50
#     repetition_penalty=1.2,
#     pad_token_id=self.tokenizer.eos_token_id,  # Explicitly set pad token
#     stopping_criteria=stopping_criteria,  # Inject custom stopping criteria
# )

# f"""
#         <think>\nTASK: Generate a single sentence summarizing a student's demographic details using the given structured JSON data.

#         INSTRUCTIONS:
#         1. Extract the demographic key-value pairs fields from the JSON:

#         2. Construct a concise, grammatically correct sentence using the JSON key-value pairs.

#         3. Ensure the sentence follows this format:

#         Example:'''Joe Shmow, student ID: 93000, age 20, Male, is a US student from Ethiopia.'''

#         4. DO NOT:
#         Generate multiple sentences.
#         Infer or hallucinate any details not present in the provided JSON.
#         Change the structure of the JSON output.

#         INPUT JSON:
#         {demographic}
#     """


def build_rag_prompt(contexts: list[dict], query: str) -> str:
    context_text = "\n".join(
        [f"- {json.dumps(ctx['metadata']['text'], indent=2)}" for ctx in contexts]
    )
    return (
        "<|im_start|>system\nYou are a helpful assistant using student records to answer questions.<|im_end|>\n"
        f"<|im_start|>user\n{query}\n\nRelevant Student Information:\n{context_text}<|im_end|>\n"
        "<|im_start|>assistant\n<think>"
    )


def get_math_prompt(prompt: str) -> str:
    return f"""
    For the problem "{prompt}", involving the expression, produce a clear, step-by-step explanation of how the result was derived. If any discrepancies are identified, re-evaluate and provide the correct steps without referencing prior mistakes or external explanations.

    Restate the problem clearly using <think> tags.
    Break down the solution into intuitive, logically ordered steps. Use plain language and avoid unnecessary technical jargon.
    Conclude the reasoning with the final answer inside <answer> tags. Do not use markdown.
    Include a <verifier_answer> section containing only the result, formatted for automated systems. Do not use markdown.
    All reasoning must be within the <think></think> tags, no reasoning should occur outside those tags

    <think>
    Problem: {prompt}
    Reasoning:
    1. Use <think> tag to start reasoning.
    2. Start with the expression.
    3. Explain and solve step by step.
    4. Conclude with the final result based on the above steps.
    5. Recheck the steps, and answer if there is a discrepancy
    6. End with </think> tag to end reasoning.
    </think>
    <answer>final answer</answer>
    <verifier_answer>clean answer used for verification</verifier_answer>
    """


def get_chat_prompt(prompt: str):
    return f"""
    A conversation between User and Assistant.
    The user asks a question, and the Assistant solves it.

    The assistant first thinks about the reasoning process in the mind and then provides the user with the answer.
    The reasoning process and answer are enclosed within <think> </think> and <answer> </answer> tags.

    <think> reasoning process here </think>

    <answer> answer here </answer>.

    User: {prompt} Assistant: <think>
    """


def get_demographic_prompt(demographic: dict) -> str:
    """
    <|reserved_special_token_93|> usually starts the "Thinking" session.
    <|reserved_special_token_94|> usually ends the "Thinking" session.
    <|reserved_special_token_81|> usually ends the return answer.
    """
    demographic_text = f"{demographic.get('first_name')} {demographic.get('middle_name')} {demographic.get('last_name')}, student ID: {demographic.get('student_id')}, age {demographic.get('current_age')}, {demographic.get('gender')}, is a {demographic.get('citizenship')} student from {demographic.get('nationality')}. The student's marital status is {demographic.get('marital_status')} and a {demographic.get('veteran_category')}"

    return f"""
        INSTRUCTIONS:
        1. Extract the demographic details from a student's EXTRACTED DATA.

        2. Construct a grammatically correct sentence using the student's EXTRACTED DATA.

        3. DO NOT hallucinate any details not present in the provided EXTRACTED DATA.

        4. Format your final answer as a grammatically correct sentence using the student's EXTRACTED DATA in this format: ```<final> insert answer here </final>```

        EXTRACTED DATA: {demographic_text}
    """


def get_semester_prompt(semester_index, semester: dict, degree_types: list) -> str:
    degree = semester.get("addmission_population")
    degree_types.append(degree)

    degree_text = f"of starting their {degree} degree"
    if semester_index > 0 and degree_types[semester_index] != degree:
        degree_text = f"of continuing their {degree} degree"
    else:
        degree_text = f"starting a new {degree} degree"

    semester_text = f"In semester {semester_index + 1} {degree_text}, the student was {semester.get('student_population')}, {semester.get('enrollment_status')}. In {semester.get('academic_period_desc')}, enrolling {semester.get('semester_time_status')}, they were on {semester.get('academic_standing')} with a GPA of {semester.get('gpa')} and a cumulative GPA of {semester.get('cgpa')}."

    courses_text = ""
    for course in semester.get("courses"):
        courses_text += f" {course.get('llm_response')} "

    # Will be too many tokens to handle for 4 GB of VRAM
    # semester_text += courses_text

    return (
        f"""
        INSTRUCTIONS:
        1. Extract the student's semester {semester_index + 1} details from the EXTRACTED DATA.

        2. Construct a grammatically correct sentence using the student's semester {semester_index + 1} EXTRACTED DATA.

        3. DO NOT hallucinate any details not present in the EXTRACTED DATA.

        4. Format your final answer as a grammatically correct sentence using the semester {semester_index + 1} EXTRACTED DATA in this format: ```<final> insert answer here </final>```

        SEMESTER {semester_index + 1} EXTRACTED DATA: {semester_text}

        <think>\nMake sure to only use details provided in EXTRACTED DATA. Review your sentence with the data before submission!
    """,
        degree_types,
    )


def get_course_prompt(course: dict) -> str:
    if course.get("course_passed_ind") == "Y":
        passed = "passed"
    elif course.get("course_failed_in") == "Y":
        passed = "failed"
    else:
        passed = "taken"

    course_text = f"The student had {passed} the {course.get('course_level')} course {course.get('course_id')} - {course.get('course_title')} covering {course.get('subject')} and with a grade value of {course.get('grade_value')}. The course was {course.get('instruction_method')}. They earned {course.get('credits_earned')}/{course.get('credits_attempted')}. They achieved a {course.get('final_grade')}."

    if course.get("is_transferred") != "N":
        course_text += " The student had transferred to this course."

    if course.get("is_withdrawn") != "N":
        course_text += " The student had withdrawn from this course."

    return f"""
        INSTRUCTIONS:
        1. Extract the student course details from a student's course EXTRACTED DATA.

        2. Construct a grammatically correct sentence using the student's course EXTRACTED DATA.

        3. DO NOT hallucinate any details not present in the provided course EXTRACTED DATA.

        4. Format your final answer as a grammatically correct sentence using the student's course EXTRACTED DATA in this format: ```<final> insert answer here </final>```

        EXTRACTED DATA: {course_text}
    """


def extract_final_content(text):
    if text is None:
        return None
    # Define regex pattern to search for <final> and extract content until </
    pattern = r"<final>(.*?)</final>"

    # Find all matches
    matches = re.findall(pattern, text, re.DOTALL)

    if matches:
        # Trim leading/trailing whitespace
        extracted = matches[-1].strip()
        logger.warning(extracted)

        # Ensure there is actual content inside <final> and not just whitespace
        if not extracted:
            # Quit if empty after trimming
            return None

        if "insert answer here" in extracted:
            return None

        if "<final>" in extracted.lower() or "</final>" in extracted.lower():
            return None

        if "final" in extracted.lower() or "extracted" in extracted.lower():
            return None

        return extracted
    else:
        # Return None if <final> is not found
        return None
