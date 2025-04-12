from util.logger import ColorLogger
from util.prompts import build_rag_prompt
from rag.retrieval import QueryEmbedder, Retriever
from rag.simple_vector_store import SimpleVectorStore


class RAGPipeline:
    def __init__(self, vector_store_dir: str, generator):
        self.logger = ColorLogger("RAGPipeline")
        self.store = SimpleVectorStore()
        self.store.load(vector_store_dir)

        self.embedder = QueryEmbedder()
        self.retriever = Retriever(self.store, self.embedder)
        self.generator = generator

    def run(self, query: str) -> str:
        self.logger.info("Running RAG pipeline...")

        contexts = self.retriever.retrieve(query)
        for ctx in contexts:
            self.logger.debug(f"Retrieved ({ctx})")

        prompt = build_rag_prompt(contexts, query)
        response = self.generator.generate_response(prompt)

        return response


# Add batch inference (run_batch(queries: List[str]))

# Support multiple prompt formats (rag.run(query, style="chain_of_thought"))

# Add caching or memoization for repeated queries
