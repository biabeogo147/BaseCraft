import json
from config import default_config
from app.vectorstore.milvus import milvus_connect
from app.model.model_query.base_ollama_query import embedding_ollama, base_query_ollama

if __name__ == "__main__":
    question = ["How is data stored in milvus?"]

    milvus_connect.init_db()
    client = milvus_connect.client
    search = client.search(
        data=embedding_ollama(text=question, model_name=default_config.MXBAI_EMBED_LARGE_MODEL_NAME),
        collection_name=default_config.RAG_GITHUB_COLLECTION,
        search_params={"metric_type": "COSINE", "params": {}},
        output_fields=["content"],
        limit=3,
    )

    retrieved_lines_with_distances = [
        (res["entity"]["text"], res["distance"]) for res in search[0] # If length(search) = length(number of questions)
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
        model_name=default_config.LLAMA_MODEL_NAME,
    )

    print(result)
