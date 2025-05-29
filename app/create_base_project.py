import os
import shutil
from app.utils.utils import check_llm_connection, check_vector_store_connection
from app.utils.generation_workflow import generate_scripts, generate_directories_and_files


def create_project(requirement: str, root_dir: str):
    root_dir = os.path.join("..\\generated_project", root_dir)
    if os.path.exists(root_dir):
        shutil.rmtree(root_dir)
    os.makedirs(root_dir)

    root_json_files = os.path.join(root_dir, "json_files")
    os.makedirs(root_json_files)

    root_source_code = os.path.join(root_dir, "source_code")
    os.makedirs(root_source_code)

    prompt = f"""I want to create a project satisfied these requirement: {requirement}."""
    generate_scripts(prompt, root_json_files)
    generate_directories_and_files(root_json_files, root_source_code)


def main():
    is_llm_connect = check_llm_connection()
    # is_cache_connect = check_cache_connection()
    is_vector_store_connect = check_vector_store_connection()

    if not is_llm_connect or not is_vector_store_connect:
        print("Failed to connect to LLM or Vector Store. Please check your configuration.")
        exit(0)

    # requirement = input("Enter the project requirement: ")
    requirement = "Generate a simple Flappy bird game using python."
    # root_dir = input("Enter the project name: ")
    root_dir = "FlappyBird"

    print(f"Creating base project for {requirement} at {root_dir}...")
    create_project(requirement, root_dir)
    print("Completed!")


if __name__ == "__main__":
    main()