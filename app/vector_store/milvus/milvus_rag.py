import json
from app.config import app_config
from app.vector_store.milvus import milvus_db
from app.config.app_config import DEFAULT_METRIC_TYPE
from app.vector_store.milvus.milvus_db import setup_vector_store
from app.llm.llm_query.base_ollama_query import embedding_ollama


def query_milvus(prompt: str, collection_name: str, limit: int = 10) -> str:
    setup_vector_store()

    client = milvus_db.client
    search = client.search(
        data=embedding_ollama(text=[prompt], model_name=app_config.MXBAI_EMBED_LARGE_MODEL_NAME),
        search_params={"metric_type": DEFAULT_METRIC_TYPE, "params": {}},
        collection_name=collection_name,
        output_fields=["content"],
        limit=limit,
    )

    retrieved_lines_with_distances = [
        (res["entity"]["content"], res["distance"]) for res in search[0] # If length(search) = length(number of questions)
    ]

    return json.dumps(retrieved_lines_with_distances, indent=4)
