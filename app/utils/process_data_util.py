import os
from typing import List
from llama_index.core.prompts import RichPromptTemplate
from app.model.model_query.base_ollama_query import embedding_ollama
from app.config.app_config import IS_OLLAMA, MXBAI_EMBED_LARGE_MODEL_NAME, EMBED_VECTOR_DIM


def prompt_template(context: str, previous_response: str, path: str) -> str:
    """
    Create a prompt template for model.
    """
    template_str = open(path, "r", encoding="utf-8").read()

    if not template_str:
        print("Prompt template is empty")
        return ""

    try:
        qa_template = RichPromptTemplate(template_str)
        prompt = qa_template.format(context_str=context, previous_response=previous_response)
    except Exception as e:
        print(f"Error adding context: {e}. \n Using default prompt.")
        prompt = template_str

    return prompt


def embedding_text(line) -> List[float]:
    """
    Embeds the input text using the specified embedding model.
    """
    if IS_OLLAMA:
        result = embedding_ollama([line], model_name=MXBAI_EMBED_LARGE_MODEL_NAME)
        return result[0]
    else:
        return [0] * EMBED_VECTOR_DIM


def is_file(path) -> bool:
    normalized_path = os.path.normpath(path)
    name = os.path.basename(normalized_path)
    _, extension = os.path.splitext(name)
    if extension:
        return True
    return False


def save(result: str, output_file: str) -> None:
    with open(output_file, "w", encoding="utf-8") as f:
        f.write(result)
    print(f"Response saved to {output_file}")
