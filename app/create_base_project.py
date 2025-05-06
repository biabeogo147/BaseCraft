import os
import json
import shutil
from app.config import default_config
from app.model.model_query.base_ollama_query import query_programming_ollama, query_idea_ollama, query_structure_ollama


def generate_project(requirement: str, root_dir: str):
    prompt = f"""I want to create a project satisfied these requirement: {requirement}."""
    system_prompt_for_idea_model = open(f"model/model_prompt/prompt_for_idea_model.txt", "r", encoding="utf-8").read()
    system_prompt_for_structure_model = open(f"model/model_prompt/prompt_for_structure_model.txt", "r", encoding="utf-8").read()
    system_prompt_for_programming_model = open(f"model/model_prompt/prompt_for_programming_model.txt", "r", encoding="utf-8").read()

    print("Generating a simple application idea...")
    idea_result = query_idea_ollama(
        prompt,
        system_prompt_for_idea_model,
        default_config.LLAMA_MODEL_NAME
    )
    with open(f"{root_dir}\\idea_model_response.json", "w", encoding="utf-8") as f:
        f.write(idea_result)

    print("Generating a simple application structure...")
    structure_result = query_structure_ollama(
        idea_result,
        system_prompt_for_structure_model,
        default_config.LLAMA_MODEL_NAME,
    )
    with open(f"{root_dir}\\structure_model_response.json", "w", encoding="utf-8") as f:
        f.write(structure_result)

    print("Generating a simple application code script...")
    programming_result = query_programming_ollama(
        structure_result,
        system_prompt_for_programming_model,
        default_config.LLAMA_MODEL_NAME,
    )
    with open(f"{root_dir}\\programming_model_response.json", "w", encoding="utf-8") as f:
        f.write(programming_result)
    print("Done querying programming Ollama")
    print("Response saved")

    if programming_result:
        try:
            structure_result = json.loads(structure_result)
            programming_result = json.loads(programming_result)
            for directory in structure_result.get("directories", []):
                os.makedirs(os.path.join(root_dir, directory), exist_ok=True)
            for file_info in programming_result.get("files", []):
                file_path = os.path.join(root_dir, file_info["path"])
                os.makedirs(os.path.dirname(file_path), exist_ok=True)
                with open(file_path, "w", encoding="utf-8") as f:
                    f.write(file_info["content"])
        except Exception as e:
            print(f"Error occur: {e}")
    else:
        print("No programming result found.")


def create_project(requirement: str, root_dir: str):
    root_dir = os.path.join("..\\generated_project", root_dir)
    if os.path.exists(root_dir):
        shutil.rmtree(root_dir)
    os.makedirs(root_dir)
    generate_project(requirement, root_dir)


def main():
    requirement = input("Enter the project requirement: ")
    # requirement = "Generate a simple Caro game using python."
    root_dir = input("Enter the project name: ")

    print(f"Creating base project for {requirement} at {root_dir}...")
    create_project(requirement, root_dir)
    print("Completed!")


if __name__ == "__main__":
    main()