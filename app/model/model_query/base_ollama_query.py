from ollama import Client
from typing import Optional, List

from pydantic.json_schema import model_json_schema

from app.config.app_config import OLLAMA_HOST
from app.model.model_output.pydantic.hierarchy_structure_schema import HierarchyDirectory
from app.model.model_output.pydantic.idea_schema import Idea
from app.model.model_output.pydantic.programming_schema import DirectoryStructure
from app.model.model_output.pydantic.structure_schema import DirectoryDescription

client = Client(
    host=OLLAMA_HOST,
)


def base_query_ollama(prompt: str, model_name: str, system_prompt: str, model_json_schema: Optional[dict]) -> str:
    try:
        response = client.generate(
            prompt=prompt,
            model=model_name,
            system=system_prompt,
            format=model_json_schema,
        )
        if 'response' in response:
            return response['response']
        else:
            print("No 'response' key in Ollama response.")
            print(response)
            return ""
    except Exception as e:
        print(f"Error connecting to Ollama: {e}")
        return ""


def query_ollama(prompt: str, model_role: str, model_name: str) -> str:
    system_prompt = open(f"../model_prompt/prompt_for_{model_role}_model.txt", "r", encoding="utf-8").read()
    schema_mapping = {
        "idea": Idea.model_json_schema,
        "structure": DirectoryDescription.model_json_schema,
        "hierarchy_structure": HierarchyDirectory.model_json_schema,
        "programming": DirectoryStructure.model_json_schema,
    }
    model_json_schema = schema_mapping.get(model_role, lambda: None)()
    result = base_query_ollama(
        prompt=prompt,
        model_name=model_name,
        system_prompt=system_prompt,
        model_json_schema=model_json_schema,
    )
    return result


def embedding_ollama(text: List[str], model_name: str) -> List[List[float]]:
    response = client.embed(
        model=model_name,
        input=text,
    )
    return response['embeddings']