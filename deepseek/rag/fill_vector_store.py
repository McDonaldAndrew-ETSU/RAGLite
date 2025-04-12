import os
import json
import torch
from simple_vector_store import SimpleVectorStore
from sentence_transformers import SentenceTransformer

DATA_DIR = "../../data/cleaned_Students"
BAAI_BGE_SMALL_EN = os.path.abspath("./embedding_models/BAAI-bge-small-en")
INFLOAT_E5_LARGE_V2 = os.path.abspath("./embedding_models/infloat-e5-large-v2")
STORE_DIR = "./simple_vector_store"


def load_student_data(file_path: str) -> tuple:
    """Load student JSON and convert to a string format."""
    with open(file_path, "r") as f:
        data = json.load(f)
    # You can customize this to be smarter later
    return json.dumps(data), data


def main():
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Using device: {device}")

    print("Loading embedding model...")
    model = SentenceTransformer(
        model_name_or_path=INFLOAT_E5_LARGE_V2,
        device=device,
        cache_folder=INFLOAT_E5_LARGE_V2,
        local_files_only=True,
    )

    store = SimpleVectorStore()

    for student_file in os.listdir(DATA_DIR):
        if student_file.endswith(".json"):
            doc_id = os.path.splitext(student_file)[0]
            file_path = os.path.join(DATA_DIR, student_file)

            text, data = load_student_data(file_path)
            embedding = model.encode(text, convert_to_tensor=True, device=device)

            store.add(
                doc_id,
                embedding,
                metadata={
                    "text": {
                        "demographic": data[0]["demographic"],
                        "semester": data[0]["semesters"][-1],
                    }
                },
            )
            print(f"Encoded and added: {doc_id}")

    print("Saving vector store...")
    store.save(STORE_DIR)
    print("Done.")


if __name__ == "__main__":
    main()
