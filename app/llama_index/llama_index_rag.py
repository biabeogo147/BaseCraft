from app.config import app_config
from app.vector_store.milvus import milvus_db
from app.llama_index import llama_index_crud_vectordb
from app.config.llama_index_config import get_llama_index_model


if __name__ == "__main__":
    question = "How many people in the kitchen?"

    milvus_db.init_db(app_config.LLAMA_INDEX_DB, app_config.LLAMA_INDEX_COLLECTION)
    if app_config.INSERT_RANDOM_DATA:
        llama_index_crud_vectordb.insert_random_data()

    llm = get_llama_index_model()
    vector, response = llama_index_crud_vectordb.query_index(question, 3, llm)
    for item in vector:
        print(f"Content: {item['content']}, Score: {item['score']}")
    print(f"Response: {response}")

