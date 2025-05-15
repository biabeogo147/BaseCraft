from redis import Redis
from app.config import app_config
from llama_index.core.llms import LLM
from redisvl.schema import IndexSchema
from llama_index.llms.ollama import Ollama
from langchain_text_splitters import Language
from llama_index.embeddings.ollama import OllamaEmbedding
from llama_index.vector_stores.redis import RedisVectorStore
from llama_index.vector_stores.milvus import MilvusVectorStore
from llama_index.core.base.embeddings.base import BaseEmbedding
from llama_index.storage.chat_store.redis import RedisChatStore
from llama_index.core.vector_stores.types import BasePydanticVectorStore

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
    if app_config.IS_OLLAMA:
        return Ollama(
            model=app_config.LLAMA_MODEL_NAME,
            base_url=app_config.OLLAMA_HOST,
        )
    return LLM()


def get_llama_index_embedding() -> BaseEmbedding:
    if app_config.IS_OLLAMA:
        return OllamaEmbedding(
            model_name=app_config.MXBAI_EMBED_LARGE_MODEL_NAME,
            base_url=app_config.OLLAMA_HOST,
        )
    return BaseEmbedding()


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
    return BasePydanticVectorStore(stores_text=True)


def get_llama_index_cache() -> BasePydanticVectorStore:
    if app_config.CACHE_NAME:
        schema =  IndexSchema.from_dict({
            "index": {
                "name": "my-index",
                "key_separator": ":",
                "storage_type": "json",
            },
            "fields": [
                {
                    "name": "id",
                    "type": "tag"
                },
                {
                    "name": "content",
                    "type": "text"
                },
                {
                    "name": "dense_vector",
                    "type": "vector",
                    "attrs": {
                        "algorithm": "flat",
                        "datatype": "float32",
                        "distance_metric": "cosine",
                        "dims": app_config.EMBED_VECTOR_DIM,
                    }
                },
                # Metadata
                {
                    "name": "repo_name",
                    "type": "tag"
                },
                {
                    "name": "language",
                    "type": "tag"
                },
                {
                    "name": "chunk_index",
                    "type": "tag"
                },
                {
                    "name": "path",
                    "type": "tag"
                },
                {
                    "name": "type",
                    "type": "tag"
                }
            ]
        })
        redis = Redis(host=app_config.REDIS_HOST, port=app_config.REDIS_PORT, password=app_config.REDIS_PASSWORD)
        return RedisVectorStore(
            redis=redis,
            schema=schema,
            overwrite=app_config.RENEW_DB,
        )
    return BasePydanticVectorStore(stores_text=True)


def get_llama_index_chat_store():
    if app_config.CACHE_NAME == "redis":
        redis = Redis(host=app_config.REDIS_HOST, port=app_config.REDIS_PORT, password=app_config.REDIS_PASSWORD)
        return RedisChatStore(redis_client=redis, ttl=3000)
    return BasePydanticVectorStore(stores_text=True)

