from app.config import app_config
from app.vector_store.milvus import milvus_db
from app.llama_index import llama_index_crud_vectordb
from llama_index.core.prompts import RichPromptTemplate
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

    template_str = """We have provided context information below.
    ---------------------
    {{ context_str }}
    ---------------------
    Given this information, please answer the question: {{ query_str }}
    """
    qa_template = RichPromptTemplate(template_str)
    prompt = qa_template.format(context_str=vector, query_str=question)
    messages = qa_template.format_messages(context_str=vector, query_str=question)
    print(f"Response: {response}")
    print(f"Prompt: {prompt}")
    print(f"Messages: {messages}")