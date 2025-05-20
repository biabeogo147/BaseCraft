import os
import json
import shutil
from app.utils.generating_workflow import generate_script


def generate_project(requirement: str, root_dir: str):
    prompt = f"""I want to create a project satisfied these requirement: {requirement}."""
    idea_result, structure_result, programming_results = generate_script(prompt, root_dir)
    if programming_results:
        try:
            structure_result = json.loads(structure_result)
            for directory in structure_result.get("directories", []):
                os.makedirs(os.path.join(root_dir, directory), exist_ok=True)
            for programming_result in programming_results:
                programming_result_json = json.loads(programming_result)
                for file_info in programming_result_json.get("files", []):
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