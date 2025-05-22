from llama_index.core.prompts import RichPromptTemplate
from app.llama_index.llama_index_vectordb import query_index
from app.config.llama_index_config import get_llama_index_model
from app.config.app_config import GITHUB_RAW_CODE_COLLECTION, LLAMA_MODEL_NAME

if __name__ == "__main__":
    question = "Create Python Flappy Bird project"

    llm = get_llama_index_model(model_name=LLAMA_MODEL_NAME)
    vector, response = query_index(question, 3, GITHUB_RAW_CODE_COLLECTION, llm)
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