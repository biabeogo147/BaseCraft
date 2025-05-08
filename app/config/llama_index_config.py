from app.config import app_config
from llama_index.core.llms import LLM
from llama_index.llms.ollama import Ollama
from llama_index.embeddings.ollama import OllamaEmbedding
from llama_index.vector_stores.milvus import MilvusVectorStore
from llama_index.core.base.embeddings.base import BaseEmbedding
from llama_index.core.vector_stores.types import BasePydanticVectorStore

_ollama_model =  Ollama(
    model=app_config.LLAMA_MODEL_NAME,
    base_url=app_config.OLLAMA_HOST,
)

_ollama_embedding = OllamaEmbedding(
    model_name=app_config.MXBAI_EMBED_LARGE_MODEL_NAME,
    base_url=app_config.OLLAMA_HOST,
)

_milvus_vector_store = MilvusVectorStore(
    uri=app_config.MILVUS_HOST,
    token=f"{app_config.MILVUS_USER}:{app_config.MILVUS_PASSWORD}",
    collection_name=app_config.RAG_GITHUB_COLLECTION,
    dim=app_config.EMBED_VECTOR_DIM,
    enable_sparse=False,
    metric_type="COSINE"
)


def get_llama_index_model() -> LLM:
    if app_config.IS_OLLAMA:
        return _ollama_model
    return _ollama_model


def get_llama_index_embedding() -> BaseEmbedding:
    if app_config.IS_OLLAMA:
        return _ollama_embedding
    return _ollama_embedding


def get_llama_index_vector_store() -> BasePydanticVectorStore:
    if app_config.VECTORDB_NAME == "milvus":
        return _milvus_vector_store
    return _milvus_vector_store

