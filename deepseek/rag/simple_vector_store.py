import os
import json
import torch
from typing import List, Dict, Optional, Tuple


class SimpleVectorStore:
    def __init__(self):
        self.doc_store: Dict[str, torch.Tensor] = {}
        self.meta_store: Dict[str, dict] = {}

    def add(self, doc_id: str, vector: torch.Tensor, metadata: Optional[dict] = None):
        if not isinstance(vector, torch.Tensor):
            raise TypeError("Vector must be a torch.Tensor")
        self.doc_store[doc_id] = vector.detach().cpu()
        if metadata:
            self.meta_store[doc_id] = metadata

    def search(
        self, query_vector: torch.Tensor, top_k: int = 5
    ) -> List[Tuple[str, float]]:
        query_vector = query_vector.detach().cpu()
        all_ids = list(self.doc_store.keys())
        all_vectors = torch.stack([self.doc_store[doc_id] for doc_id in all_ids])

        # Compute cosine similarity
        dot = torch.matmul(all_vectors, query_vector)
        norms = all_vectors.norm(dim=1) * query_vector.norm()
        scores = dot / norms

        top_k_indices = torch.topk(scores, k=top_k).indices.tolist()
        results = [(all_ids[i], scores[i].item()) for i in top_k_indices]
        return results

    def get(self, doc_id: str) -> Optional[torch.Tensor]:
        return self.doc_store.get(doc_id, None)

    def get_metadata(self, doc_id: str) -> Optional[dict]:
        return self.meta_store.get(doc_id, None)

    def save(self, dir_path: str):
        os.makedirs(dir_path, exist_ok=True)

        # Save vectors
        all_ids = list(self.doc_store.keys())
        all_vectors = torch.stack([self.doc_store[doc_id] for doc_id in all_ids])
        torch.save(all_vectors, os.path.join(dir_path, "vectors.pt"))

        # Save IDs and metadata
        with open(os.path.join(dir_path, "doc_ids.json"), "w") as f:
            json.dump(all_ids, f)
        with open(os.path.join(dir_path, "metadata.json"), "w") as f:
            json.dump(self.meta_store, f)

    def load(self, dir_path: str):
        vectors = torch.load(os.path.join(dir_path, "vectors.pt"))
        with open(os.path.join(dir_path, "doc_ids.json")) as f:
            doc_ids = json.load(f)
        with open(os.path.join(dir_path, "metadata.json")) as f:
            self.meta_store = json.load(f)

        self.doc_store = {doc_id: vectors[i] for i, doc_id in enumerate(doc_ids)}
