from redisvl.schema import IndexSchema
from app.config.app_config import EMBED_VECTOR_DIM, RAG_GITHUB_COLLECTION


def create_redis_github_schema() -> IndexSchema:
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