from llama_index.core.llms import LLM
from llama_index.core.schema import BaseNode
from app.vector_store.milvus import milvus_db
from typing import List, Sequence, Optional, Tuple
from llama_index.core import VectorStoreIndex, Document
from llama_index.core.ingestion import IngestionPipeline
from app.config.app_config import LLAMA_INDEX_DB, LLAMA_INDEX_COLLECTION, RENEW_COLLECTION, INSERT_RANDOM_DATA
from app.config.llama_index_config import get_llama_index_embedding, get_llama_index_vector_store

embedding = get_llama_index_embedding()
vector_store = get_llama_index_vector_store()


def setup_vector_store():
    milvus_db.init_db(LLAMA_INDEX_DB, LLAMA_INDEX_COLLECTION)
    if RENEW_COLLECTION:
        milvus_db.drop_collection(LLAMA_INDEX_COLLECTION)
        milvus_db.create_collection(LLAMA_INDEX_COLLECTION)
    if INSERT_RANDOM_DATA:
        insert_random_data()


def insert_nodes_from_documents(documents: List[Document]) -> Sequence[BaseNode]:
    """Create nodes from documents."""
    try:
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


def insert_random_data():
    text = [
        "There are 2 people in the kitchen.",
        "There are 5 people in the bathroom.",
        "There are 10 people in the living room.",
        "Van Nhan is a Dau Buoi."
    ]
    documents = [Document(text=text_line) for i, text_line in enumerate(text)]
    insert_nodes_from_documents(documents)
    print("Inserted random data into the vector store.")


def query_index(query_text: str, top_k: int, llm: Optional[LLM] = None) -> Tuple[List[dict], str]:
    try:
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