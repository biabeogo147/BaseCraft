import os
from typing import List
from app.llm.llm_output.idea_schema import Idea
from llama_index.core.prompts import RichPromptTemplate
from app.llm.llm_output.programming_schema import File
from app.llm.llm_query.base_ollama_query import embedding_ollama, ollama_query
from app.llm.llm_output.hierarchy_structure_schema import FileRequirements
from app.llm.llm_output.description_structure_schema import FileDescriptions, FileDescription
from app.config.app_config import IS_OLLAMA, MXBAI_EMBED_LARGE_MODEL_NAME, EMBED_VECTOR_DIM


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


def embedding_text(line) -> List[float]:
    """
    Embeds the input text using the specified embedding llm.
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


def llm_query(prompt: str, model_name: str, countSelfLoop: int = 0, context: str = "", model_role: str = None) -> str:
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
        "file_description": FileDescription.model_json_schema,
        "idea_summary": Idea.model_json_schema,
    }
    model_json_schema = schema_mapping.get(model_role, lambda: None)()

    response = None
    while countSelfLoop:
        system_prompt = prompt_template(
            context=context,
            path=f"llm/model_prompt/prompt_for_{model_role}_model.txt",
            previous_response= "" if response is None else response.get('response', ''),
        )
        if IS_OLLAMA:
            response = ollama_query(
                prompt=prompt,
                model_name=model_name,
                system_prompt=system_prompt,
                model_json_schema=model_json_schema,
            )
        else:
            response = {'response': 'This is a mock response for testing purposes.'}
        countSelfLoop -= 1

    if 'response' in response:
        return response['response']
    else:
        print("No 'response' key in response.")
        print(response)
        return ""
