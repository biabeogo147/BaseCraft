import os
import shutil
from app.utils.generating_workflow import generate_scripts, generate_directories_and_files


def create_project(requirement: str, root_dir: str):
    root_dir = os.path.join("..\\generated_project", root_dir)
    if os.path.exists(root_dir):
        shutil.rmtree(root_dir)
    os.makedirs(root_dir)

    root_json_files = os.path.join(root_dir, "json_files")
    prompt = f"""I want to create a project satisfied these requirement: {requirement}."""
    generate_scripts(prompt, root_json_files)
    generate_directories_and_files(root_json_files, root_dir)


def main():
    requirement = input("Enter the project requirement: ")
    # requirement = "Generate a simple Caro game using python."
    root_dir = input("Enter the project name: ")

    print(f"Creating base project for {requirement} at {root_dir}...")
    create_project(requirement, root_dir)
    print("Completed!")


if __name__ == "__main__":
    main()