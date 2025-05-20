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

_model = {}
_cache = {}
_embedding = {}
_vector_store = {}


def get_llama_index_model(model_name: str) -> LLM:
    if IS_OLLAMA:
        if _model.get("OLLAMA").get(model_name) is None:
            _model["OLLAMA"][model_name] = Ollama(
                model=model_name,
                base_url=OLLAMA_HOST,
            )
        return _model[model_name]
    return LLM()


def get_llama_index_embedding(embedding_name: str) -> BaseEmbedding:
    if IS_OLLAMA:
        if _embedding.get("OLLAMA").get(embedding_name) is None:
            _embedding["OLLAMA"][embedding_name] = OllamaEmbedding(
                model_name=embedding_name,
                base_url=OLLAMA_HOST,
            )
        return _embedding["OLLAMA"][embedding_name]
    return BaseEmbedding()


def get_llama_index_vector_store(collection_name: str) -> BasePydanticVectorStore:
    if VECTORDB_NAME == "milvus":
        if _vector_store.get(VECTORDB_NAME).get(collection_name) is None:
            _vector_store[VECTORDB_NAME][collection_name] = MilvusVectorStore(
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
        return _vector_store[VECTORDB_NAME][collection_name]
    return BasePydanticVectorStore(stores_text=True)


def get_llama_index_cache(db_number: int) -> BaseKVStore:
    if CACHE_NAME == "redis":
        if _cache.get(CACHE_NAME).get(db_number) is None:
            redis_client = Redis(
                db=db_number,
                host=REDIS_HOST,
                port=REDIS_PORT,
                password=REDIS_PASSWORD,
            )
            _cache[CACHE_NAME][db_number] = RedisCache.from_redis_client(redis_client)
            if RENEW_CACHE:
                redis_client.flushdb()
        return _cache[CACHE_NAME][db_number]
    return BaseKVStore()