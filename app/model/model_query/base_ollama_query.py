from ollama import Client
from typing import List, Optional
from app.config.app_config import OLLAMA_HOST
from app.model.model_output.idea_schema import Idea
from app.utils.process_data_util import prompt_template
from app.model.model_output.programming_schema import File
from app.model.model_output.hierarchy_structure_schema import FileRequirements
from app.model.model_output.description_structure_schema import FileDescriptions, FileDescription

client = Client(
    host=OLLAMA_HOST,
)


def base_query_ollama(prompt: str, model_name: str, countSelfLoop: int = 0, context: Optional[str] = None, model_role: Optional[str] = None) -> str:
    try:
        schema_mapping = {
            # generation workflow
            "idea": Idea.model_json_schema,
            "description_structure": FileDescriptions.model_json_schema,
            "hierarchy_structure": FileRequirements.model_json_schema,
            "programming": File.model_json_schema,

            # repo process workflow
            "file_description": FileDescription.model_json_schema,
            "idea_summary": Idea.model_json_schema,
        }
        model_json_schema = schema_mapping.get(model_role, lambda: None)()

        response = None
        while countSelfLoop:
            system_prompt = prompt_template(
                context=context,
                path=f"model/model_prompt/prompt_for_{model_role}_model.txt",
                previous_response= "" if response is None else response.get('response', ''),
            )
            response = client.generate(
                prompt=prompt,
                model=model_name,
                system=system_prompt,
                format=model_json_schema,
            )
            countSelfLoop -= 1

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