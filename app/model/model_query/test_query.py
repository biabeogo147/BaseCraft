import os
from app.config import app_config
from app.model.model_query.base_ollama_query import query_idea_ollama, query_structure_ollama, query_programming_ollama

if __name__ == "__main__":
    root_dir = "response"
    if not os.path.exists(root_dir):
        os.makedirs(root_dir)

    system_prompt_for_idea_model = open(f"../model_prompt/prompt_for_idea_model.txt", "r", encoding="utf-8").read()
    system_prompt_for_structure_model = open(f"../model_prompt/prompt_for_structure_model.txt", "r", encoding="utf-8").read()
    system_prompt_for_programming_model = open(f"../model_prompt/prompt_for_programming_model.txt", "r", encoding="utf-8").read()

    # Test the query_idea_ollama function
    prompt = "Generate a simple Caro game using python."
    print("Generating a simple application idea...")
    idea_result = query_idea_ollama(prompt, system_prompt_for_idea_model, default_config.LLAMA_MODEL_NAME)
    with open(f"{root_dir}\\idea_model_response.json", "w", encoding="utf-8") as f:
        f.write(idea_result)

    # Test the query_structure_ollama function
    print("Generating a simple application structure...")
    structure_result = query_structure_ollama(idea_result, system_prompt_for_structure_model, default_config.LLAMA_MODEL_NAME)
    with open(f"{root_dir}\\structure_model_response.json", "w", encoding="utf-8") as f:
        f.write(structure_result)

    # Test the query_programming_ollama function
    print("Generating a simple application code script...")
    programming = query_programming_ollama(structure_result, system_prompt_for_programming_model, default_config.LLAMA_MODEL_NAME)
    with open(f"{root_dir}\\programming_model_response.json", "w", encoding="utf-8") as f:
        f.write(programming)