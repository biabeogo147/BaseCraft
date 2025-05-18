from typing import List
from app.model.model_query.base_ollama_query import embedding_ollama
from app.config.app_config import IS_OLLAMA, MXBAI_EMBED_LARGE_MODEL_NAME, EMBED_VECTOR_DIM


def emb_text(line) -> List[float]:
    if IS_OLLAMA:
        result = embedding_ollama([line], model_name=MXBAI_EMBED_LARGE_MODEL_NAME)
        return result[0]
    else:
        return [0] * EMBED_VECTOR_DIM