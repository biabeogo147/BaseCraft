import json
from app.config import app_config
from app.vector_store.milvus import milvus_db
from app.model.model_query.base_ollama_query import embedding_ollama, base_query_ollama

if __name__ == "__main__":
    question = ["Who is Van Nhan?"]

    if app_config.RENEW_DB:
        milvus_db.drop_github_db()
    milvus_db.init_db()
    if app_config.INSERT_RANDOM_DATA:
        milvus_db.insert_random_data()

    client = milvus_db.client

    search = client.search(
        data=embedding_ollama(text=question, model_name=app_config.MXBAI_EMBED_LARGE_MODEL_NAME),
        collection_name=app_config.RAG_GITHUB_COLLECTION,
        search_params={"metric_type": "COSINE", "params": {}},
        output_fields=["content"],
        limit=3,
    )

    retrieved_lines_with_distances = [
        (res["entity"]["content"], res["distance"]) for res in search[0] # If length(search) = length(number of questions)
    ]
    print(json.dumps(retrieved_lines_with_distances, indent=4))
    context = "\n".join(
        [line_with_distance[0] for line_with_distance in retrieved_lines_with_distances]
    )

    SYSTEM_PROMPT = """
    Human: You are an AI assistant. You are able to find answers to the questions from the contextual passage snippets provided.
    """

    USER_PROMPT = f"""
    Use the following pieces of information enclosed in <context> tags to provide an answer to the question enclosed in <question> tags.
    <context>
    {context}
    </context>
    
    <question>
    {question}
    </question>
    """

    result = base_query_ollama(
        prompt=USER_PROMPT,
        model_json_schema=None,
        system_prompt=SYSTEM_PROMPT,
        model_name=app_config.LLAMA_MODEL_NAME,
    )

    print(result)
