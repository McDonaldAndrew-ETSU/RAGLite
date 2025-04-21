import re
import json
import evaluate
from deepreason import DeepReasonGenerator


class Eval:
    def __init__(self):
        self.bleu = evaluate.load("bleu")
        self.rouge = evaluate.load("rouge")

    def extract_prompt_and_reference(self, text):
        # 1. Extract the prompt
        user_match = re.search(
            r"<\|im_start\|>user\n(.*?)<\|im_end\|>", text, re.DOTALL
        )
        prompt = user_match.group(1).strip() if user_match else ""

        # 2. Extract assistant content (full block)
        assistant_match = re.search(
            r"<\|im_start\|>assistant\n.*?<\|im_end\|>", text, re.DOTALL
        )
        reference = assistant_match.group(0).strip() if assistant_match else ""

        return prompt, reference

    def evaluate_response(self, reference, generated):
        bleu_score = self.bleu.compute(
            predictions=[generated], references=[[reference]]
        )
        rouge_score = self.rouge.compute(
            predictions=[generated], references=[reference]
        )
        return {
            "BLEU": bleu_score["bleu"],
            "ROUGE-1": rouge_score["rouge1"],
            "ROUGE-2": rouge_score["rouge2"],
            "ROUGE-L": rouge_score["rougeL"],
        }

    def run(self, model):
        llm = DeepReasonGenerator(
            model_name_or_path=f"./models/{model}",
            already_quantized=False,
        )
        input_file = "../data/evaluation_data/acad_merge_eval.jsonl"
        output_file = f"../data/evaluation_data/evaluation_results_{model}.jsonl"

        # Process Coldstart Entries
        results = []
        counter = 0

        with open(input_file, "r") as f:
            for line in f:
                obj = json.loads(line)
                prompt, reference = self.extract_prompt_and_reference(obj["text"])

                if not prompt or not reference:
                    continue

                generated = llm.generate_response(prompt)
                scores = self.evaluate_response(reference, generated)
                results.append(
                    {
                        "prompt": prompt,
                        "reference": reference,
                        "generated": generated,
                        "scores": scores,
                    }
                )
                counter += 1
                print(f"Line {counter} complete")

        # Save Results
        with open(output_file, "w") as f:
            for r in results:
                f.write(json.dumps(r) + "\n")

        print(f"Saved {len(results)} evaluations to {output_file}")


if __name__ == "__main__":
    eval = Eval()

    base_model = input("Enter the input for the base model:")
    trained_model = "Enter the input for the trained model:"
    eval.run(model=base_model)
    eval.run(model=trained_model)
