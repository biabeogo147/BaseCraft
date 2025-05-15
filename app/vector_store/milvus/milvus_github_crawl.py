from app.utils import splitter
from langchain_text_splitters import Language
from app.vector_store.milvus import milvus_db
from app.utils.github_crawl import get_files_on_repo, github
from app.vector_store.milvus.milvus_db import setup_vector_store
from app.config.app_config import GITHUB_API_KEY, REPO_NAMES, RAG_GITHUB_COLLECTION

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
                data.append({
                    "dense_vector": milvus_db.emb_text(chunk),
                    "content": chunk,
                    # Metadata
                    "type": file['type'],
                    "path": file['path'],
                    "repo_name": repo_name,
                })
            milvus_db.insert_data(
                collection_name=RAG_GITHUB_COLLECTION,
                data=data
            )
        print(f"Finished processing repository: {repo_name}")