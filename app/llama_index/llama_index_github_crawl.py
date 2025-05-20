from app.utils import splitter
from llama_index.core import Document
from langchain_text_splitters import Language
from app.utils.github_crawl import github, get_files_on_repo
from app.config.llama_index_config import LANGUAGE_LLAMA_INDEX
from app.config.app_config import GITHUB_API_KEY, REPO_NAMES, REDIS_GITHUB_RAG_DB, RAG_GITHUB_COLLECTION
from app.llama_index.llama_index_vectordb import insert_nodes_to_vector_store_from_documents, insert_nodes_to_cache

if __name__ == "__main__":
    if GITHUB_API_KEY is None:
        raise ValueError("GITHUB_API_KEY environment variable not set")

    repos = REPO_NAMES
    for repo_name in repos:
        print(f"Processing repository: {repo_name}")
        repo = github.get_repo(repo_name)
        files = get_files_on_repo(repo)
        print(f"Total files: {len(files)}")

        for file in files:
            data_vector_store = []
            chunks = splitter.split_source_code(file['content'], LANGUAGE_LLAMA_INDEX.get(file['language'], Language.PYTHON)) if file['content'] else []
            for index, chunk in enumerate(chunks):
                document = Document(
                    text=chunk,
                    metadata={
                        "type": file['type'],
                        "path": file['path'],
                        "chunk_index": index,
                        "repo_name": repo_name,
                        "language": file['language'],
                    }
                )
                data_vector_store.append(document)
            nodes = insert_nodes_to_vector_store_from_documents(RAG_GITHUB_COLLECTION, data_vector_store)
            # insert_nodes_to_cache(REDIS_GITHUB_RAG_DB, nodes)
        print(f"Finished processing repository: {repo_name}")