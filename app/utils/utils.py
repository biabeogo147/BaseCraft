import os
from typing import List
from app.llm.llm_output.idea_schema import Idea
from app.llm.llm_output.programming_schema import File
from llama_index.core.prompts import RichPromptTemplate
from llama_index.core.base.llms.types import ChatMessage
from app.llm.llm_output.hierarchy_structure_schema import FileRequirements
from app.llm.llm_output.description_structure_schema import FileDescriptions
from app.llm.llm_query.base_ollama_query import embedding_ollama, ollama_query
from app.config.llama_index_config import get_llama_index_model, get_llama_index_embedding
from app.config.app_config import API_PROVIDER, MXBAI_EMBED_LARGE_MODEL_NAME, EMBED_VECTOR_DIM, IS_LLAMA_INDEX, \
    API_PROVIDER_EMBEDDING


def prompt_template(context: str, previous_response: str, path: str) -> str:
    """
    Create a prompt template for llm.
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


def embedding_text(line: str) -> List[float]:
    """
    Embeds the input text using the specified embedding llm.
    """
    if IS_LLAMA_INDEX:
        embedding_llm = get_llama_index_embedding(embedding_name=MXBAI_EMBED_LARGE_MODEL_NAME)
        result = embedding_llm.get_text_embedding(line)
        if isinstance(result, list) and len(result) == EMBED_VECTOR_DIM:
            return result
    else:
        if API_PROVIDER_EMBEDDING == "ollama":
            result = embedding_ollama([line], model_name=MXBAI_EMBED_LARGE_MODEL_NAME)
            return result[0]
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


def llm_query(prompt: str, model_name: str, count_self_loop: int = 0, context: str = "", model_role: str = None) -> str:
    """
    Query the LLM with the given prompt and parameters.
    """
    schema_mapping = {
        # generation workflow
        "idea": Idea.model_json_schema,
        "description_structure": FileDescriptions.model_json_schema,
        "hierarchy_structure": FileRequirements.model_json_schema,
        "programming": File.model_json_schema,
        "compile_error_fix": File.model_json_schema,

        # repo process workflow
        "file_description": None,
        "idea_summary": Idea.model_json_schema,
    }
    schema_method = schema_mapping.get(model_role)
    model_json_schema = schema_method() if callable(schema_method) else None

    response = ""
    while count_self_loop:
        count_self_loop -= 1
        system_prompt = prompt_template(
            context=context,
            previous_response= response,
            path=f"llm/llm_prompt/prompt_for_{model_role}_model.txt",
        )
        if IS_LLAMA_INDEX:
            llm = get_llama_index_model(model_name=model_name)
            if model_json_schema is None:
                response_llama_index = llm.chat(
                    [ChatMessage(role="user", content=prompt),
                     ChatMessage(role="system", content=system_prompt)],
                )
            else:
                structure_llm = llm.as_structured_llm(output_cls=model_json_schema)
                response_llama_index = structure_llm.chat(
                    [ChatMessage(role="user", content=prompt),
                     ChatMessage(role="system", content=system_prompt)],
                )

            if hasattr(response_llama_index, "response") and hasattr(response_llama_index.message, "content"):
                response = response_llama_index.message.content
            else:
                response = 'No content in response from Llama Index.'
                count_self_loop = 0

        else:
            if API_PROVIDER == "ollama":
                response_llm = ollama_query(
                    prompt=prompt,
                    model_name=model_name,
                    system_prompt=system_prompt,
                    model_json_schema=model_json_schema,
                )
            else:
                response_llm = {'response': 'This is a mock response for testing purposes.'}

            if 'response' in response_llm:
                response = response_llm['response']
            else:
                response = 'No response key in response dictionary.'
                count_self_loop = 0

    return response