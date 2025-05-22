from app.config import app_config
from app.vector_store.milvus import milvus_db
from langchain_text_splitters import Language
from app.vector_store.redis.redis_db import setup_cache
from app.utils.process_data_util import split_source_code, get_import_list, process_raw_hierarchy
from app.vector_store.milvus.milvus_db import setup_vector_store
from app.model.model_query.base_ollama_query import base_query_ollama
from app.utils.process_data_util import get_files_on_repo, embedding_text, get_github_connect
from app.config.app_config import REPO_NAMES, GITHUB_RAW_CODE_COLLECTION, GITHUB_DESCRIPTION_STRUCTURE_COLLECTION, \
    DEFAULT_TEXT_FIELD, DEFAULT_EMBEDDING_FIELD, GITHUB_HIERARCHY_STRUCTURE_COLLECTION

if __name__ == "__main__":
    setup_vector_store()
    cache_indexes = setup_cache()
    github = get_github_connect()

    repos = REPO_NAMES
    for repo_name in repos:
        print(f"Processing repository: {repo_name}")
        repo = github.get_repo(repo_name)
        files = get_files_on_repo(repo)
        print(f"Total files: {len(files)}")

        file_hierarchy = []
        file_descriptions = []
        for file in files:
            raw_source_code = []
            chunks = split_source_code(file['content'], Language.PYTHON) if file['content'] else []
            for i, chunk in enumerate(chunks):
                embedding = embedding_text(chunk)
                raw_source_code.append({
                    DEFAULT_TEXT_FIELD: chunk,
                    DEFAULT_EMBEDDING_FIELD: embedding,
                    # Metadata
                    "chunk_index": i,
                    "type": file['type'],
                    "path": file['path'],
                    "repo_name": repo_name,
                    "language": file['language'],
                })
            milvus_db.insert_data(
                collection_name=GITHUB_RAW_CODE_COLLECTION,
                data=raw_source_code,
            )

            description_file = base_query_ollama(
                countSelfLoop=2,
                prompt=file['content'],
                model_role="file_description_from_code",
                model_name=app_config.LLAMA_MODEL_NAME,
            )
            embedding_description = embedding_text(description_file)
            file_descriptions.append({
                DEFAULT_TEXT_FIELD: description_file,
                DEFAULT_EMBEDDING_FIELD: embedding_description,
                # Metadata
                "type": file['type'],
                "path": file['path'],
                "repo_name": repo_name,
                "language": file['language'],
            })

            import_list = get_import_list(file['content'])
            file_hierarchy.append({
                "path": file['path'],
                "depend_on": import_list,
            })

        milvus_db.insert_data(
            collection_name=GITHUB_DESCRIPTION_STRUCTURE_COLLECTION,
            data=file_descriptions,
        )

        file_hierarchy = process_raw_hierarchy(file_hierarchy)
        milvus_db.insert_data(
            collection_name=GITHUB_HIERARCHY_STRUCTURE_COLLECTION,
            data=[],
        )

        print(f"Finished processing repository: {repo_name}")