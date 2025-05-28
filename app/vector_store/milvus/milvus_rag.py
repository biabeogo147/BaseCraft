from app.config import app_config
from typing import Dict, Any, List
from app.vector_store.milvus import milvus_db
from app.llm.llm_query.base_ollama_query import embedding_ollama
from app.config.app_config import DEFAULT_METRIC_TYPE, KNOWLEDGE_BASE_DB, DEFAULT_EMBEDDING_FIELD, DEFAULT_TEXT_FIELD


def query_milvus_with_prompt(
        prompt: str,
        collection_name: str,
        output_fields: List[str],
        limit: int = 10) -> List[Dict[str, Any]]:

    output_fields = [DEFAULT_TEXT_FIELD] + output_fields
    client = milvus_db.get_client_instance()
    client.use_database(KNOWLEDGE_BASE_DB)

    search = client.search(
        data=embedding_ollama(text=[prompt], model_name=app_config.MXBAI_EMBED_LARGE_MODEL_NAME),
        search_params={"metric_type": DEFAULT_METRIC_TYPE, "params": {}},
        anns_field=DEFAULT_EMBEDDING_FIELD,
        collection_name=collection_name,
        output_fields=output_fields,
        limit=limit,
    )

    retrieved_lines_with_distances = [
        {
            "distance": res["distance"],
            **{field: res["entity"][field] for field in output_fields}
        }
        for res in search[0]  # If length(search) = length(number of questions)
    ]

    return retrieved_lines_with_distances


def query_milvus_with_metadata(
        metadata: Dict[str, Any],
        collection_name: str,
        output_fields: List[str],
        limit: int = 100) -> List[Dict[str, Any]]:

    output_fields = [DEFAULT_TEXT_FIELD] + output_fields
    client = milvus_db.get_client_instance()
    client.use_database(KNOWLEDGE_BASE_DB)

    filter_expr = " AND ".join(
        [f"{key} == '{value}'" for key, value in metadata.items()]
    )
    result = client.query(
        collection_name=collection_name,
        output_fields=output_fields,
        filter_params=metadata,
        filter=filter_expr,
        limit=limit,
    )

    return result
