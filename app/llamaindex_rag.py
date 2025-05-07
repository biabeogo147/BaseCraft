from llama_index.core import VectorStoreIndex, SimpleDirectoryReader, PromptTemplate
from llama_index.vector_stores.milvus import MilvusVectorStore
from llama_index.embeddings.ollama import OllamaEmbedding
from llama_index.llms.ollama import Ollama
from config import default_config

if __name__ == "__main__":
    question = "How is data stored in milvus?"

    embed_model = OllamaEmbedding(
        model_name=default_config.MXBAI_EMBED_LARGE_MODEL_NAME,
        base_url=default_config.OLLAMA_HOST,
    )

    vector_store = MilvusVectorStore(
        uri=default_config.MILVUS_HOST,
        collection_name=default_config.RAG_GITHUB_COLLECTION,
        dim=default_config.EMBED_VECTOR_DIM,
        similarity_metric="COSINE",
    )

    index = VectorStoreIndex.from_vector_store(
        vector_store=vector_store,
        embed_model=embed_model,
    )

    llm = Ollama(
        model=default_config.LLAMA_MODEL_NAME,
        base_url=default_config.OLLAMA_HOST,
    )

    query_engine = index.as_query_engine(
        llm=llm,
    )

    response = query_engine.query(question)

    print(response.response)