from typing import Dict
from redis import Redis
from redisvl.index import SearchIndex
from redisvl.schema import IndexSchema
from app.config.app_config import EMBED_VECTOR_DIM, GITHUB_RAW_CODE_COLLECTION, REDIS_HOST, REDIS_PORT, REDIS_PASSWORD, \
    RENEW_CACHE, REDIS_GITHUB_DB, REDIS_USER_PROJECT_DB, USER_RAW_CODE_COLLECTION, DEFAULT_METRIC_TYPE

_index = {}
_redis = {}


def setup_cache() -> Dict[int, SearchIndex]:
    """
    Set up the SearchIndex for Redis.
    """
    global _redis
    if len(_redis) == 0:
        schemas = [
            (REDIS_GITHUB_DB, create_redis_github_schema()),
            (REDIS_USER_PROJECT_DB, create_redis_user_project_schema())
        ]
        for schema in schemas:
            _redis[schema[0]] = Redis(
                db=schema[0],
                host=REDIS_HOST,
                port=REDIS_PORT,
                password=REDIS_PASSWORD
            )
            print(f"Redis {schema[0]} connection established: {_redis[schema[0]].ping()}")
            if RENEW_CACHE:
                _redis[schema[0]].flushdb()
                print(f"Redis cache {schema[0]} flushed.")
            _index[schema[0]] = create_redis_index(schema[1], db_number=schema[0])
            print(f"Redis search index {schema[0]} created.")
        print("Vector store setup complete.")
        return _index
    else:
        print("Redis connection already established.")
        return _index


def create_redis_index(schema: IndexSchema, db_number: int) -> SearchIndex:
    """
    Create a Redis index with the given schema.
    """
    index = SearchIndex(
        schema=schema,
        validate_on_load=True,
        redis_client=_redis[db_number],
    )
    index.create(overwrite=True)
    return index


def create_redis_github_schema() -> IndexSchema:
    """
    Create a Redis schema for the GitHub knowledge base.
    """
    schema = IndexSchema.from_dict({
        "index": {
            "name": GITHUB_RAW_CODE_COLLECTION,
            "prefix": "github",
            "key_separator": ":",
            "storage_type": "hash",
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
                    "dims": EMBED_VECTOR_DIM,
                    "distance_metric": DEFAULT_METRIC_TYPE.lower(),
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


def create_redis_user_project_schema() -> IndexSchema:
    """
    Create a Redis schema for the user project knowledge base.
    """
    schema = IndexSchema.from_dict({
        "index": {
            "name": USER_RAW_CODE_COLLECTION,
            "prefix": "user_project",
            "key_separator": ":",
            "storage_type": "hash",
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
                    "dims": EMBED_VECTOR_DIM,
                    "distance_metric": DEFAULT_METRIC_TYPE.lower(),
                }
            },
            # Metadata
            {
                "name": "project_name",
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

