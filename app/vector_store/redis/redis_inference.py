from redis import Redis
from redisvl.index import SearchIndex
from app.vector_store.redis.redis_db import create_redis_github_schema
from app.config.app_config import REDIS_HOST, REDIS_PORT, REDIS_PASSWORD, EMBED_VECTOR_DIM


def insert_sample():
    redis = Redis(host=REDIS_HOST, port=REDIS_PORT, password=REDIS_PASSWORD)
    print("Connected to Redis: ", redis.ping())

    index = SearchIndex(
        schema=create_redis_github_schema(),
        redis_client=redis,
        validate_on_load=True
    )
    index.create(overwrite=True, drop=False)
    index.load(
        data=[{
            "id": "1",
            "doc_id": "1",
            "text": "hello world",
            "vector": [0 for _ in range(EMBED_VECTOR_DIM)],
            "repo_name": "test_repo",
            "language": "python",
            "chunk_index": 0,
            "path": "/path/to/file.py",
            "type": "file",
        }],
    )
    index.delete(drop=True)


if __name__ == "__main__":
    insert_sample()