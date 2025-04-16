import os
import time
import json
import torch
from util.logger import ColorLogger
from rag.rag_pipeline import RAGPipeline
from transformers import (
    AutoTokenizer,
    BitsAndBytesConfig,
    AutoModelForCausalLM,
    StoppingCriteriaList,
)

# from util.prompts import (
#     get_chat_prompt,
#     get_math_prompt,
#     build_rag_prompt,
#     get_course_prompt,
#     get_semester_prompt,
#     extract_final_content,
#     get_demographic_prompt,
# )
from util.stream_stop import StreamStoppingCriteria


logger = ColorLogger(__name__)
llama_8b = "./models/DeepSeek-R1-Distill-Llama-8B"
qwen_7B = "./models/DeepSeek-R1-Distill-Qwen-7B"
qwen_1_5B = "./models/DeepSeek-R1-Distill-Qwen-1.5B"
cached = "./models/currently_cached_model"
trained = "./models/qwen_lora_trained"


class DeepReasonGenerator:
    def __init__(self, model_name_or_path: str, already_quantized: bool = True) -> None:
        start = time.time()
        logger.info("Initializing model...")
        if already_quantized:
            os.environ["HF_HUB_OFFLINE"] = "1"
            self.model = AutoModelForCausalLM.from_pretrained(
                model_name_or_path, local_files_only=True, trust_remote_code=True
            )

            self.tokenizer = AutoTokenizer.from_pretrained(
                model_name_or_path, local_files_only=True, trust_remote_code=True
            )

            self.device = "cuda" if torch.cuda.is_available() else "cpu"
            self.model.to(self.device)
        else:
            self.bnb_cfg = BitsAndBytesConfig(
                load_in_4bit=True, bnb_4bit_compute_dtype=torch.float16
            )
            logger.info("Created configurations")

            self.model = AutoModelForCausalLM.from_pretrained(
                model_name_or_path,
                trust_remote_code=True,
                torch_dtype=torch.float16,
                quantization_config=self.bnb_cfg,
            )
            logger.info("Loaded model")

            self.tokenizer = AutoTokenizer.from_pretrained(
                model_name_or_path, use_fast=True, trust_remote_code=True
            )
            logger.info("Loaded tokenizer")

            self.model.save_pretrained("./models/currently_cached_model")
            self.tokenizer.save_pretrained("./models/currently_cached_model")
            logger.info("Model and Tokenizer saved to cache directory")
        end = time.time()
        elapsed = end - start
        logger.info(f"Initialized in {elapsed:.2f} seconds - Awaiting prompt")

    def generate_response(self, prompt):
        start = time.time()
        logger.info(f"Received prompt!")

        input_data = self.tokenizer(
            f"{prompt}",
            return_tensors="pt",
            padding=True,  # Ensures padding tokens are handled correctly
            truncation=True,  # Truncates if input is too long
            max_length=4096,  # Adjust based on VRAM limits
        )
        input_ids = input_data.input_ids.to(self.model.device)
        # Ensures padding is ignored
        attention_mask = input_data.attention_mask.to(self.model.device)
        # Dynamically calculates tokens to avoid out-of-memory errors
        available_tokens = 4096 - input_ids.shape[1]

        # Store StreamStoppingCriteria separately
        stream_criteria = StreamStoppingCriteria(self.tokenizer)
        stopping_criteria = StoppingCriteriaList([stream_criteria])

        output_ids = self.model.generate(
            input_ids,
            attention_mask=attention_mask,  # Passes attention mask to avoid issues
            max_new_tokens=available_tokens,
            temperature=0.6,  # recommended by deep seek = 0.6, my settings = 0.55
            top_p=1.0,  # default = 1.0, my settings = 0.87
            top_k=50,  # default = 50, my settings = 41
            repetition_penalty=1.2,
            pad_token_id=self.tokenizer.eos_token_id,  # Explicitly set pad token
            stopping_criteria=stopping_criteria,  # Inject custom stopping criteria
        )

        print("\n")  # Newline after completion
        response = self.tokenizer.decode(output_ids[0], skip_special_tokens=False)

        end = time.time()
        elapsed = end - start
        logger.info(f"Generated in {elapsed:.2f} seconds")

        avg_tps, t = stream_criteria.compute_tps()
        logger.info(
            f"Avg. Tokens Per Second: {avg_tps:.2f} - Total Response Tokens: {t}"
        )

        return response


# def update_data_with_llm_responses(json_filepath, llm):
#     data = None
#     with open(json_filepath, "r", encoding="utf-8") as f:
#         data = json.load(f)

#     data["demographic"] = update_demographic_data(data, llm)

#     data["semesters"] = update_semesters_data(data, llm)

#     # Write to updated
#     with open(f"../data/updated_data.json", "w", encoding="utf-8") as f:
#         json.dump(data, f, indent=4)
#         logger.info("finished writing!")


# def update_demographic_data(data, llm):
#     demographic_data = data["demographic"]
#     demographic_prompt = get_demographic_prompt(demographic_data)
#     extracted = None
#     while extracted is None:
#         extracted = extract_final_content(llm.generate_response(demographic_prompt))

#     demographic_data["llm_response"] = extracted
#     return demographic_data


# def update_semesters_data(data, llm):
#     semesters_data = data["semesters"]
#     degree_types = []
#     for i, semester in enumerate(semesters_data):
#         courses = semester.get("courses")

#         semester["courses"] = update_courses_data(courses, llm)

#         semester_prompt, degree_types = get_semester_prompt(i, semester, degree_types)
#         extracted = None
#         while extracted is None:
#             extracted = extract_final_content(llm.generate_response(semester_prompt))

#         semester["llm_response"] = extracted

#     return semesters_data


# def update_courses_data(courses_data, llm):
#     for course in courses_data:
#         course_prompt = get_course_prompt(course)
#         extracted = None
#         while extracted is None:
#             extracted = extract_final_content(llm.generate_response(course_prompt))

#         course["llm_response"] = extracted

#     return courses_data


if __name__ == "__main__":
    llm = DeepReasonGenerator(model_name_or_path=trained, already_quantized=False)
    rag = RAGPipeline(vector_store_dir="./rag/simple_vector_store", generator=llm)

    while True:
        user_input = input("Rag Query? (Y/N): ")
        if user_input == "Y":
            query = input("Rag Query: ")
            response = rag.run(query)
        elif user_input == "N":
            prompt = input("You: ")
            response = llm.generate_response(prompt)
        elif user_input.lower() in ["exit", "quit"]:
            logger.info("Exiting chat. Have a good day!")
            break
