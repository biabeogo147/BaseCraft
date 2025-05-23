from typing import List, Dict, Any
from ollama import Client, GenerateResponse
from app.config.app_config import OLLAMA_HOST

client = Client(
    host=OLLAMA_HOST,
)


def ollama_query(prompt: str, model_name: str, system_prompt: str, model_json_schema: Dict[str, Any]) -> GenerateResponse:
    try:
        response = client.generate(
            prompt=prompt,
            model=model_name,
            system=system_prompt,
            format=model_json_schema,
        )
        return response
    except Exception as e:
        print(f"Error connecting to Ollama: {e}")
        return GenerateResponse(response=f"Error connecting to Ollama: {e}")


def embedding_ollama(text: List[str], model_name: str) -> List[List[float]]:
    response = client.embed(
        model=model_name,
        input=text,
    )
    return response['embeddings']