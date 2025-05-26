from redis import Redis
from app.config.app_config import *
from llama_index.core.llms import LLM
from llama_cloud import GeminiEmbedding
from llama_index.llms.ollama import Ollama
from llama_index.llms.google_genai import GoogleGenAI
from llama_index.embeddings.ollama import OllamaEmbedding
from llama_index.vector_stores.milvus import MilvusVectorStore
from llama_index.core.storage.kvstore.types import BaseKVStore
from llama_index.core.base.embeddings.base import BaseEmbedding
from llama_index.core.vector_stores.types import BasePydanticVectorStore
from llama_index.storage.kvstore.redis import RedisKVStore as RedisCache

LLAMA_INDEX_DB = "default"

_model = {}
_cache = {}
_embedding = {}
_vector_store = {}


def get_llama_index_model(model_name: str) -> LLM:
    if _model.get(API_PROVIDER) is None:
        _model[API_PROVIDER] = {}

    if _model[API_PROVIDER].get(model_name) is None:
        if API_PROVIDER == "ollama":
            _model[API_PROVIDER][model_name] = Ollama(
                model=model_name,
                base_url=OLLAMA_HOST,
            )
        elif API_PROVIDER == "gemini":
            _model[API_PROVIDER][model_name] = GoogleGenAI(
                model=model_name,
                api_key=GOOGLE_API_KEY,
            )
    return _model[API_PROVIDER][model_name]


def get_llama_index_embedding(embedding_name: str) -> BaseEmbedding:
    if _embedding.get(API_PROVIDER_EMBEDDING) is None:
        _embedding[API_PROVIDER_EMBEDDING] = {}

    if _embedding[API_PROVIDER_EMBEDDING].get(embedding_name) is None:
        if API_PROVIDER_EMBEDDING == "ollama":
            _embedding[API_PROVIDER_EMBEDDING][embedding_name] = OllamaEmbedding(
                base_url=OLLAMA_HOST,
                model_name=embedding_name,
            )
        elif API_PROVIDER_EMBEDDING == "gemini":
            _embedding[API_PROVIDER_EMBEDDING][embedding_name] = GeminiEmbedding(
                api_key=GOOGLE_API_KEY,
                model_name=embedding_name,
            )
    return _embedding[API_PROVIDER_EMBEDDING][embedding_name]


def get_llama_index_vector_store(collection_name: str) -> BasePydanticVectorStore:
    if _vector_store.get(VECTORDB_NAME) is None:
        _vector_store[VECTORDB_NAME] = {}

    if VECTORDB_NAME == "milvus":
        if _vector_store[VECTORDB_NAME].get(collection_name) is None:
            _vector_store[VECTORDB_NAME][collection_name] = MilvusVectorStore(
                uri=MILVUS_HOST,
                overwrite=RENEW_DB,
                enable_sparse=False,
                dim=EMBED_VECTOR_DIM,
                text_key=DEFAULT_TEXT_FIELD,
                collection_name=collection_name,
                embedding_key=DEFAULT_EMBEDDING_FIELD,
                similarity_metric=DEFAULT_METRIC_TYPE,
                token=f"{MILVUS_USER}:{MILVUS_PASSWORD}",
            )
        return _vector_store[VECTORDB_NAME][collection_name]
    return BasePydanticVectorStore(stores_text=True)


def get_llama_index_cache(db_number: int) -> BaseKVStore:
    if _cache.get(CACHE_NAME) is None:
        _cache[CACHE_NAME] = {}

    if CACHE_NAME == "redis":
        if _cache[CACHE_NAME].get(db_number) is None:
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