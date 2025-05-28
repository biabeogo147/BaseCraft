import json
from typing import Dict, Any
from app.config import app_config
from app.vector_store.milvus import milvus_db
from app.config.app_config import DEFAULT_METRIC_TYPE, KNOWLEDGE_BASE_DB
from app.llm.llm_query.base_ollama_query import embedding_ollama


def query_milvus_with_prompt(prompt: str, collection_name: str, limit: int = 10) -> str:
    client = milvus_db.get_client_instance()
    client.use_database(KNOWLEDGE_BASE_DB)

    search = client.search(
        data=embedding_ollama(text=[prompt], model_name=app_config.MXBAI_EMBED_LARGE_MODEL_NAME),
        search_params={"metric_type": DEFAULT_METRIC_TYPE, "params": {}},
        collection_name=collection_name,
        output_fields=["content"],
        limit=limit,
    )

    # Extracting content, distance and metadata from the search results
    retrieved_lines_with_distances = [
        (res["entity"]["content"], res["distance"]) for res in search[0] # If length(search) = length(number of questions)
    ]

    return json.dumps(retrieved_lines_with_distances, indent=4)


def query_milvus_with_metadata(metadata: Dict[str, Any], collection_name: str, limit: int = 100) -> str:
    client = milvus_db.get_client_instance()
    client.use_database(KNOWLEDGE_BASE_DB)

    filter_expr = " AND ".join(
        [f"{key} == '{value}'" for key, value in metadata.items()]
    )
    search = client.search(
        search_params={"metric_type": DEFAULT_METRIC_TYPE, "params": {}},
        data=[[0.0 for _ in range(app_config.EMBED_VECTOR_DIM)]],
        collection_name=collection_name,
        output_fields=["content"],
        filter_params=metadata,
        filter=filter_expr,
        limit=limit,
    )

    retrieved_lines_with_distances = [
        (res["entity"]["content"]) for res in search[0] # If length(search) = length(number of questions)
    ]

    return json.dumps(retrieved_lines_with_distances, indent=4)
