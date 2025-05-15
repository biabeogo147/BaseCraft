from app.utils import splitter
from llama_index.core import Document
from langchain_text_splitters import Language
from app.utils.github_crawl import github, get_files_on_repo
from app.config.app_config import GITHUB_API_KEY, REPO_NAMES
from app.llama_index.llama_index_vectordb import setup_vector_store, insert_nodes_from_documents


if __name__ == "__main__":
    if GITHUB_API_KEY is None:
        raise ValueError("GITHUB_API_KEY environment variable not set")

    setup_vector_store()

    repos = REPO_NAMES
    for repo_name in repos:
        print(f"Processing repository: {repo_name}")
        repo = github.get_repo(repo_name)
        files = get_files_on_repo(repo)
        print(f"Total files: {len(files)}")

        for file in files:
            data = []
            chunks = splitter.split_source_code(file['content'], Language.PYTHON) if file['content'] else []
            for chunk in chunks:
                document = Document(
                    text=chunk,
                    metadata={
                        "type": file['type'],
                        "path": file['path'],
                        "repo_name": repo_name,
                    }
                )
                data.append(document)
            insert_nodes_from_documents(data)
        print(f"Finished processing repository: {repo_name}")