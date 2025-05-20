from typing import List
from ollama import Client
from app.config.app_config import OLLAMA_HOST
from app.model.model_output.idea_schema import Idea
from app.model.model_output.programming_schema import File
from app.model.model_output.hierarchy_structure_schema import DirectoryHierarchy
from app.model.model_output.description_structure_schema import DirectoryDescription

client = Client(
    host=OLLAMA_HOST,
)


def base_query_ollama(prompt: str, model_role: str, model_name: str) -> str:
    try:
        system_prompt = open(f"../model_prompt/prompt_for_{model_role}_model.txt", "r", encoding="utf-8").read()
        schema_mapping = {
            "idea": Idea.model_json_schema,
            "description_structure": DirectoryDescription.model_json_schema,
            "hierarchy_structure": DirectoryHierarchy.model_json_schema,
            "programming": File.model_json_schema,
        }
        model_json_schema = schema_mapping.get(model_role, lambda: None)()
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


def embedding_ollama(text: List[str], model_name: str) -> List[List[float]]:
    response = client.embed(
        model=model_name,
        input=text,
    )
    return response['embeddings']