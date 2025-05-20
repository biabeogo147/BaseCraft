from redis import Redis
from app.config.app_config import *
from llama_index.core.llms import LLM
from llama_index.llms.ollama import Ollama
from langchain_text_splitters import Language
from llama_index.embeddings.ollama import OllamaEmbedding
from llama_index.vector_stores.milvus import MilvusVectorStore
from llama_index.core.storage.kvstore.types import BaseKVStore
from llama_index.core.base.embeddings.base import BaseEmbedding
from app.vector_store.milvus.milvus_db import insert_random_data
from llama_index.core.vector_stores.types import BasePydanticVectorStore
from llama_index.storage.kvstore.redis import RedisKVStore as RedisCache

LLAMA_INDEX_DB = "default"
LANGUAGE_LLAMA_INDEX = {
    "C": Language.C,
    "Go": Language.GO,
    "PHP": Language.PHP,
    "C++": Language.CPP,
    "C#": Language.CSHARP,
    "Java": Language.JAVA,
    "Ruby": Language.RUBY,
    "Rust": Language.RUST,
    "HTML": Language.HTML,
    "Swift": Language.SWIFT,
    "Python": Language.PYTHON,
    "Kotlin": Language.KOTLIN,
    "JavaScript": Language.JS,
    "TypeScript": Language.TS,
    "Markdown": Language.MARKDOWN,
    "Shell Script": Language.POWERSHELL,
}


def get_llama_index_model() -> LLM:
    if IS_OLLAMA:
        return Ollama(
            model=LLAMA_MODEL_NAME,
            base_url=OLLAMA_HOST,
        )
    return LLM()


def get_llama_index_embedding() -> BaseEmbedding:
    if IS_OLLAMA:
        return OllamaEmbedding(
            model_name=MXBAI_EMBED_LARGE_MODEL_NAME,
            base_url=OLLAMA_HOST,
        )
    return BaseEmbedding()


def get_llama_index_vector_store(collection_name: str) -> BasePydanticVectorStore:
    if VECTORDB_NAME == "milvus":
        vector_store = MilvusVectorStore(
            uri=MILVUS_HOST,
            overwrite=RENEW_DB,
            enable_sparse=False,
            dim=EMBED_VECTOR_DIM,
            similarity_metric="COSINE",
            text_key=DEFAULT_TEXT_FIELD,
            collection_name=collection_name,
            embedding_key=DEFAULT_EMBEDDING_FIELD,
            token=f"{MILVUS_USER}:{MILVUS_PASSWORD}",
        )
        if INSERT_RANDOM_DATA:
            insert_random_data(LLAMA_INDEX_DB, collection_name)
        return vector_store
    return BasePydanticVectorStore(stores_text=True)


def get_llama_index_cache() -> BaseKVStore:
    if CACHE_NAME == "redis":
        redis_client = Redis(
            host=REDIS_HOST,
            port=REDIS_PORT,
            db=REDIS_GITHUB_RAG_DB,
            password=REDIS_PASSWORD,
        )
        if RENEW_CACHE:
            redis_client.flushdb()
        return RedisCache.from_redis_client(redis_client)
    return BaseKVStore()