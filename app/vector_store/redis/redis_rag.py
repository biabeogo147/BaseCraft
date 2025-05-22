import numpy as np
from redisvl.query import VectorQuery
from app.config.app_config import INSERT_RANDOM_DATA
from app.vector_store.redis.redis_db import setup_cache

indexes = setup_cache()


def insert_sample():
    print("Loading sample data into Redis...")

    chunk_index = [0, 1, 2, 3, 4]
    language = ["Python", "Java", "JavaScript", "C++", "C"]
    type = ["code", "image", "archive", "binary", "document"]
    repo_name = ["repo1", "repo2", "repo3", "repo4", "repo5"]
    text = [
        "There are two people in the kitchen.",
        "The cat is on the table.",
        "The dog is barking outside.",
        "The sun is shining brightly.",
        "The flowers are blooming in the garden.",
    ]
    path = [
        "path/to/file1.py",
        "path/to/file2.java",
        "path/to/file3.js",
        "path/to/file4.cpp",
        "path/to/file5.c",
    ]

    indexes[0].load(
        data=[{
                "id": f"{repo_name[i]}:{path[i]}:{chunk_index[i]}",
                "doc_id": f"{repo_name[i]}:{path[i]}",
                "vector": np.array(emb_text(text[i]), dtype=np.float32).tobytes(), # If storage type is "hash", then the vector field must be a byte array
                "text": text[i],
                # Metadata
                "type": type[i],
                "path": path[i],
                "language": language[i],
                "repo_name": repo_name[i],
                "chunk_index": chunk_index[i],
            }
            for i in range(len(text))
        ],
    )

    print("Sample data loaded into Redis.")


def query_sample():
    print("Querying Redis for similar vectors...")

    query_text = "How many people are in the kitchen?"
    query_vector = np.array(emb_text(query_text), dtype=np.float32).tobytes()

    query = VectorQuery(
        num_results=5,
        vector=query_vector,
        vector_field_name="vector",
        return_fields=["vector_distance", "id", "text"],
    )

    # Work only on Hash storage type
    results = [index.query(query=query) for index in indexes]

    print("Query results:")
    for result in results:
        for item in result:
            print(f"ID: {item['id']}, Distance: {item['vector_distance']}, Text: {item['text']}")


if __name__ == "__main__":
    if INSERT_RANDOM_DATA:
        insert_sample()
    query_sample()