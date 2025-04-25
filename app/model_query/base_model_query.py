from ollama import Client
from app.config.default_config import HOST
from app.model_output.programming_model_output import DirectoryStructure

client = Client(
    host=HOST,
)

def base_query_ollama(prompt: str, model_name: str, system_prompt: str, model_json_chema) -> str:
    try:
        response = client.generate(
            prompt=prompt,
            model=model_name,
            format=DirectoryStructure.model_json_schema(),
            system=open(f"model_prompt/{system_prompt}", "r", encoding="utf-8").read(),
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

def query_programming_ollama(prompt: str, model_name: str) -> str:
    try:
        response = client.generate(
            prompt=prompt,
            model=model_name, 
            format=DirectoryStructure.model_json_schema(),
            system=open("model_prompt/prompt_for_programming_model.txt", "r", encoding="utf-8").read(),
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