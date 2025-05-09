from ollama import Client
from typing import Optional, List
from app.config.app_config import OLLAMA_HOST
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


def query_idea_ollama(prompt: str, system_prompt: str, model_name: str) -> str:
    result = base_query_ollama(
        prompt=prompt,
        model_name=model_name,
        system_prompt=system_prompt,
        model_json_schema=Idea.model_json_schema(),
    )
    # print("Idea model response:", result)
    return result


def query_structure_ollama(prompt: str, system_prompt: str, model_name: str) -> str:
    result = base_query_ollama(
        prompt=prompt,
        model_name=model_name,
        system_prompt=system_prompt,
        model_json_schema=DirectoryDescription.model_json_schema(),
    )
    # print("Structure model response:", result)
    return result


def query_programming_ollama(prompt: str, system_prompt: str, model_name: str) -> str:
    result = base_query_ollama(
        prompt=prompt,
        model_name=model_name,
        system_prompt=system_prompt,
        model_json_schema=DirectoryStructure.model_json_schema(),
    )
    # print("Programming model response:", result)
    return result


def embedding_ollama(text: List[str], model_name: str) -> List[List[float]]:
    response = client.embed(
        model=model_name,
        input=text,
    )
    return response['embeddings']