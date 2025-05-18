from redis import Redis
from redisvl.index import SearchIndex
from redisvl.schema import IndexSchema
from app.config.app_config import EMBED_VECTOR_DIM, RAG_GITHUB_COLLECTION, REDIS_HOST, REDIS_PORT, REDIS_PASSWORD, \
    RENEW_DB

redis = Redis(host=REDIS_HOST, port=REDIS_PORT, password=REDIS_PASSWORD)
print("Connected to Redis: ", redis.ping())


def setup_cache():
    """
    Set up the vector store for Redis.
    """
    if RENEW_DB:
        redis.flushdb()
    schema = create_redis_github_schema()
    index = create_redis_index(schema)
    print("Vector store setup complete.")
    return index


def create_redis_index(schema: IndexSchema) -> SearchIndex:
    """
    Create a Redis index with the given schema.
    """
    index = SearchIndex(
        schema=schema,
        redis_client=redis,
        validate_on_load=True
    )
    index.create(overwrite=True, drop=False)
    return index


def create_redis_github_schema() -> IndexSchema:
    """
    Create a Redis schema for the GitHub knowledge base.
    """
    schema = IndexSchema.from_dict({
        "index": {
            "name": RAG_GITHUB_COLLECTION,
            "key_separator": ":",
            "storage_type": "json",
        },
        "fields": [
            {
                "name": "id",
                "type": "tag",
            },
            {
                "name": "doc_id",
                "type": "tag",
            },
            {
                "name": "text",
                "type": "text",
            },
            {
                "name": "vector",
                "type": "vector",
                "attrs": {
                    "algorithm": "flat",
                    "datatype": "float32",
                    "distance_metric": "cosine",
                    "dims": EMBED_VECTOR_DIM,
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
    return schema