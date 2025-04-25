import os
import json
import shutil
from typing import Dict
from app.config.default_config import LLAMA_MODEL_NAME
from app.model_query.base_model_query import query_programming_ollama

def generate_project_structure(project_type: str, root_dir: str) -> Dict:
    prompt = f"""I want to create a base project for {project_type}."""
    response_text = query_programming_ollama(prompt, LLAMA_MODEL_NAME)
    print("Done querying programming Ollama")
    if response_text:
        try:
            with open(f"{root_dir}/ollama_response.json", "w", encoding="utf-8") as f:
                f.write(response_text)
            print("Response saved")
            return json.loads(response_text)
        except json.JSONDecodeError:
            print("The response from Ollama is not valid JSON.")
            # print(response_text)
            return {}
    return {}

def create_project(project_type: str, root_dir: str):
    root_dir = os.path.join("..\\generated_project", root_dir)
    if os.path.exists(root_dir):
        shutil.rmtree(root_dir)
    os.makedirs(root_dir)
    
    structure = generate_project_structure(project_type, root_dir)
    # structure = json.loads(open("ollama_response.json", "r", encoding="utf-8").read())
    if not structure:
        print("Unable to generate project structure.")
        return
    
    for directory in structure.get("directories", []):
        dir_path = os.path.join(root_dir, directory)
        os.makedirs(dir_path, exist_ok=True)
        print(f"Created directory: {dir_path}")
    
    for file_info in structure.get("files", []):
        file_path = os.path.join(root_dir, file_info["path"])
        os.makedirs(os.path.dirname(file_path), exist_ok=True)
        with open(file_path, "w", encoding="utf-8") as f:
            f.write(file_info["content"])
        print(f"Created file: {file_path}")

def main():
    project_type = input("Enter the project type (e.g., Python Flask, Node.js, React): ")
    root_dir = input("Enter the target directory path (e.g., my_project): ")
    
    print(f"Creating base project for {project_type} at {root_dir}...")
    create_project(project_type, root_dir)
    print("Completed!")

if __name__ == "__main__":
    main()