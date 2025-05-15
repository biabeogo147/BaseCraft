from llama_index.core.prompts import RichPromptTemplate
from app.config.llama_index_config import get_llama_index_model
from app.llama_index.llama_index_vectordb import query_index, setup_vector_store

if __name__ == "__main__":
    question = "How many people in the kitchen?"

    setup_vector_store()

    llm = get_llama_index_model()
    vector, response = query_index(question, 3, llm)
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