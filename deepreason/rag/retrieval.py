import os
import torch
from rag.simple_vector_store import SimpleVectorStore
from sentence_transformers import SentenceTransformer

BAAI_BGE_SMALL_EN = os.path.abspath("./rag/embedding_models/BAAI-bge-small-en")
INFLOAT_E5_LARGE_V2 = os.path.abspath("./rag/embedding_models/infloat-e5-large-v2")


class QueryEmbedder:
    def __init__(self, model_path=INFLOAT_E5_LARGE_V2):
        self.model = SentenceTransformer(model_path, local_files_only=True)

    def embed(self, text: str) -> torch.Tensor:
        embedding = self.model.encode(text, convert_to_tensor=True)
        return embedding


class Retriever:
    def __init__(self, vector_store: SimpleVectorStore, embedder: QueryEmbedder):
        self.store = vector_store
        self.embedder = embedder

    def retrieve(self, query: str, top_k: int = 1) -> list[dict]:
        query_vec = self.embedder.embed(query)
        hits = self.store.search(query_vec, top_k)
        return [
            {
                "doc_id": doc_id,
                "score": score,
                "metadata": self.store.get_metadata(doc_id),
            }
            for doc_id, score in hits
        ]
