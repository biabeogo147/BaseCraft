from app.config import app_config
from llama_index.core.llms import LLM
from llama_index.llms.ollama import Ollama
from llama_index.embeddings.ollama import OllamaEmbedding
from llama_index.vector_stores.milvus import MilvusVectorStore
from llama_index.core.base.embeddings.base import BaseEmbedding
from llama_index.core.vector_stores.types import BasePydanticVectorStore


def get_llama_index_model() -> LLM:
    if app_config.IS_OLLAMA:
        return Ollama(
            model=app_config.LLAMA_MODEL_NAME,
            base_url=app_config.OLLAMA_HOST,
        )

    return Ollama(
        model=app_config.LLAMA_MODEL_NAME,
        base_url=app_config.OLLAMA_HOST,
    )


def get_llama_index_embedding() -> BaseEmbedding:
    if app_config.IS_OLLAMA:
        return OllamaEmbedding(
            model_name=app_config.MXBAI_EMBED_LARGE_MODEL_NAME,
            base_url=app_config.OLLAMA_HOST,
        )

    return OllamaEmbedding(
        model_name=app_config.MXBAI_EMBED_LARGE_MODEL_NAME,
        base_url=app_config.OLLAMA_HOST,
    )


def get_llama_index_vector_store() -> BasePydanticVectorStore:
    if app_config.VECTORDB_NAME == "milvus":
        return MilvusVectorStore(
            uri=app_config.MILVUS_HOST,
            token=f"{app_config.MILVUS_USER}:{app_config.MILVUS_PASSWORD}",
            collection_name=app_config.RAG_GITHUB_COLLECTION,
            dim=app_config.EMBED_VECTOR_DIM,
            overwrite=app_config.RENEW_DB,
            enable_sparse=False,
            metric_type="COSINE",
        )

    return MilvusVectorStore(
        uri=app_config.MILVUS_HOST,
        token=f"{app_config.MILVUS_USER}:{app_config.MILVUS_PASSWORD}",
        collection_name=app_config.RAG_GITHUB_COLLECTION,
        dim=app_config.EMBED_VECTOR_DIM,
        overwrite=app_config.RENEW_DB,
        enable_sparse=False,
        metric_type="COSINE"
    )

