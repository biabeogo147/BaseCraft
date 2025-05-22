from app.vector_store.milvus import milvus_db
from langchain_text_splitters import Language
from app.vector_store.redis.redis_db import setup_cache
from app.utils.process_data_util import split_source_code
from app.vector_store.milvus.milvus_db import setup_vector_store
from app.utils.process_data_util import get_files_on_repo, emb_text, get_github_connect
from app.config.app_config import GITHUB_API_KEY, REPO_NAMES, GITHUB_COLLECTION, REDIS_GITHUB_DB

if __name__ == "__main__":
    if GITHUB_API_KEY is None:
        raise ValueError("GITHUB_API_KEY environment variable not set")

    setup_vector_store()
    cache_indexes = setup_cache()

    repos = REPO_NAMES
    for repo_name in repos:
        print(f"Processing repository: {repo_name}")
        repo = get_github_connect().get_repo(repo_name)
        files = get_files_on_repo(repo)
        print(f"Total files: {len(files)}")

        for file in files:
            data_vector_store, data_cache = [], []
            chunks = split_source_code(file['content'], Language.PYTHON) if file['content'] else []
            for chunk in chunks:
                embedding = emb_text(chunk)
                data_vector_store.append({
                    "content": chunk,
                    "dense_vector": embedding,
                    # Metadata
                    "type": file['type'],
                    "path": file['path'],
                    "repo_name": repo_name,
                })
                data_cache.append({
                    "id": f"{repo_name}:{file['path']}:{chunks.index(chunk)}",
                    "doc_id": f"{repo_name}:{file['path']}",
                    "vector": embedding,
                    "text": chunk,
                    # Metadata
                    "path": file['path'],
                    "type": file['type'],
                    "repo_name": repo_name,
                    "language": file['language'],
                    "chunk_index": chunks.index(chunk),
                })
            milvus_db.insert_data(
                collection_name=GITHUB_COLLECTION,
                data=data_vector_store
            )
            cache_indexes[REDIS_GITHUB_DB].load(data=data_cache)

        print(f"Finished processing repository: {repo_name}")