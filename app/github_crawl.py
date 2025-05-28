from app.config.app_config import REPO_NAMES
from app.vector_store.milvus.milvus_db import setup_vector_store
from app.utils.utils import check_llm_connection, check_vector_store_connection
from app.utils.repo_process_workflow import insert_raw_code_to_vector_store, insert_file_requirements_to_vector_store, \
    insert_file_descriptions_to_vector_store, insert_idea_to_vector_store, get_files_on_repo, get_github_connect

if __name__ == "__main__":
    is_llm_connect = check_llm_connection()
    # is_cache_connect = check_cache_connection()
    is_vector_store_connect = check_vector_store_connection()

    if not is_llm_connect or not is_vector_store_connect:
        print("Failed to connect to LLM or Vector Store. Please check your configuration.")
        exit(0)

    setup_vector_store()
    github = get_github_connect()

    repos = REPO_NAMES
    for repo_name in repos:
        print(f"Processing repository: {repo_name}")
        repo = github.get_repo(repo_name)
        files = get_files_on_repo(repo)
        print(f"Total files: {len(files)}")

        insert_raw_code_to_vector_store(repo_name=repo_name, repo_files=files)
        insert_file_requirements_to_vector_store(repo_name=repo_name, repo_files=files)
        file_descriptions = insert_file_descriptions_to_vector_store(repo_name=repo_name, repo_files=files)
        insert_idea_to_vector_store(repo_name=repo_name, file_descriptions=file_descriptions)

        print(f"Finished processing repository: {repo_name}")