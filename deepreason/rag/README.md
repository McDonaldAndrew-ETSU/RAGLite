# `rag` directory

This directory represents a lightweight and portable RAG Pipeline. The following files and their directories are listed with their descriptions in the flow of the RAG Pipeline.

1. `simple_vector_store.py` - Contains the _SimpleVectorStore_ class that represents the vector store for RAG that uses PyTorch Tensor objects to represent each vector.

- Its search uses Cosine Similarity to evaluate the closest "matching" vectors for retrieval.

2. `embedding_models` directory - A directory that contains downloaded and pretrained vector models. Use for embedding data as well as user queries.

3. `fill_vector_store.py` - A script to populate a `simple_vector_store` directory, embedding the files within the top level `data` directory into a `vectors.pt`.

4. `retrieval.py` - Contains the _Retriever_ and _QueryEmbedder_ classes that embeds the user query and retrieves the closest vectors.

5. `rag_pipeline.py` - Contains the _RAGPipeline_ class that builds an appropriate prompt (from `./util/prompts.py`) using a Retrieved vector based on a user query and calls a Generator (in this case our DeepSeek model).
