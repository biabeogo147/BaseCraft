from llama_index.core.llms import LLM
from llama_index.core.schema import BaseNode
from typing import List, Sequence, Optional, Tuple
from llama_index.core import VectorStoreIndex, Document
from llama_index.core.ingestion import IngestionPipeline, IngestionCache
from app.config.app_config import RAG_GITHUB_COLLECTION, MXBAI_EMBED_LARGE_MODEL_NAME
from app.config.llama_index_config import get_llama_index_embedding, get_llama_index_vector_store, get_llama_index_cache


def insert_nodes_to_vector_store_from_documents(collection_name: str, documents: List[Document]) -> Sequence[BaseNode]:
    """Create nodes from documents."""
    try:
        vector_store = get_llama_index_vector_store(collection_name=collection_name)
        embedding = get_llama_index_embedding(embedding_name=MXBAI_EMBED_LARGE_MODEL_NAME)
        pipeline = IngestionPipeline(
            transformations=[
                embedding,
            ],
            vector_store=vector_store,
        )
        nodes = pipeline.run(documents=documents)
        return nodes
    except Exception as e:
        print(f"Failed to create nodes: {e}")
        raise


def insert_nodes_to_cache_from_documents(db_number: int, documents: List[Document]) -> Sequence[BaseNode]:
    """Create nodes from documents."""
    try:
        cache = get_llama_index_cache(db_number=db_number)
        embedding = get_llama_index_embedding(embedding_name=MXBAI_EMBED_LARGE_MODEL_NAME)
        pipeline = IngestionPipeline(
            transformations=[
                embedding,
            ],
            cache=IngestionCache(
                cache=cache,
                collection=RAG_GITHUB_COLLECTION,
            ),
        )
        nodes = pipeline.run(documents=documents)
        return nodes
    except Exception as e:
        print(f"Failed to create nodes: {e}")
        raise


def insert_nodes_to_cache(db_number: int, nodes: Sequence[BaseNode]) -> None:
    """Insert nodes to cache."""
    try:
        cache = get_llama_index_cache(db_number=db_number)
        cache.put_all(
            kv_pairs=[
                (node.get_doc_id(), node.to_dict())
                for node in nodes
            ],
            collection=RAG_GITHUB_COLLECTION,
        )
    except Exception as e:
        print(f"Failed to insert nodes to cache: {e}")
        raise


def query_index(query_text: str, top_k: int, collection_name: str, llm: Optional[LLM] = None) -> Tuple[List[dict], str]:
    try:
        vector_store = get_llama_index_vector_store(collection_name=collection_name)
        embedding = get_llama_index_embedding(embedding_name=MXBAI_EMBED_LARGE_MODEL_NAME)
        index = VectorStoreIndex.from_vector_store(
            vector_store=vector_store,
            embed_model=embedding,
        )
        query_engine = index.as_query_engine(
            similarity_top_k=top_k,
            llm=llm,
        )
        response = query_engine.query(query_text)
        results = [{"content": node.node.text, "score": node.score} for node in response.source_nodes]
        return results, response.response
    except Exception as e:
        print(f"Failed to query index: {e}")
        raise