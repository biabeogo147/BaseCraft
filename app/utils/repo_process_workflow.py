import os
from github import Github
from typing import List, Dict
from app.config import app_config
from github.Repository import Repository
from app.vector_store.milvus import milvus_db
from app.utils.utils import is_file, embedding_text, llm_query
from langchain_text_splitters import Language, RecursiveCharacterTextSplitter
from app.model.model_output.description_structure_schema import FileDescription, FileDescriptions
from app.config.app_config import LANGUAGE_LANGCHAIN, DEFAULT_TEXT_FIELD, DEFAULT_EMBEDDING_FIELD, \
    GITHUB_RAW_CODE_COLLECTION, GITHUB_DESCRIPTION_STRUCTURE_COLLECTION, GITHUB_HIERARCHY_STRUCTURE_COLLECTION, \
    GITHUB_IDEA_COLLECTION, FILE_TYPE_MAPPING, EXTENSION_TO_LANGUAGE, GITHUB_API_KEY

github = None


def split_source_code(text: str, language: Language) -> List:
    """
    Splits the input text into chunks of specified size and overlap.

    Args:
        text (str): The text to be split.
        language (Language): The programming language of the text.

    Returns:
        list: A list of text chunks.
    """
    try:
        text_splitter = RecursiveCharacterTextSplitter.from_language(
            chunk_size=1000,
            chunk_overlap=0,
            language=language,
        )
        chunks = text_splitter.split_text(text)
        return chunks
    except Exception as e:
        print(f"Error splitting text: {e}")
        return []


def get_files_on_repo(repo: Repository) -> List[Dict]:
    """
    Get all files in a GitHub repository.
    """
    files = []
    contents = repo.get_contents("")

    while contents:
        content = contents.pop(0)
        if content.type == "dir":
            contents.extend(repo.get_contents(content.path))
        else:
            path = content.path
            extension = os.path.splitext(path)[1].lower()
            file_type = FILE_TYPE_MAPPING.get(extension, 'other')
            extension = EXTENSION_TO_LANGUAGE.get(extension, 'other')

            try:
                file_content = content.decoded_content.decode("utf-8")
                files.append({"path": path, "content": file_content, "type": file_type, "language": extension})
            except Exception as e:
                print(f"Cannot decode {path}: {e}")
                files.append({"path": path, "content": None, "type": file_type, "language": extension})

    return files


def get_github_connect() -> Github:
    """
    Connect to GitHub using the provided API key.

    Returns:
        Github: A Github instance.
    """
    global github
    if github is None:
        if GITHUB_API_KEY is None:
            raise ValueError("GITHUB_API_KEY environment variable not set")
        github = Github(GITHUB_API_KEY)
    return github


def get_import_list(code: str) -> List[str]:
    """
    Extract import statements from the given code.
    """
    import_list = []
    lines = code.split("\n")
    for line in lines:
        if line.startswith("import") or line.startswith("from"):
            import_list.append(line.strip())
    return import_list


def process_raw_hierarchy(hierarchy: List[Dict]) -> List[Dict]:
    """
    Process the raw hierarchy data to ensure all dependencies are valid.
    """
    processed_hierarchy = []
    for file in hierarchy:
        if is_file(file["path"]):
            processed_hierarchy.append(file)
    return processed_hierarchy


def insert_raw_code_to_vector_store(repo_name: str, repo_files: List[Dict]):
    """
    Insert raw code into the vector store.
    """
    for file in repo_files:
        raw_source_code = []
        chunks = split_source_code(file['content'], LANGUAGE_LANGCHAIN.get(file['language'], Language.PYTHON)) \
            if file['content'] else []
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


def insert_file_descriptions_to_vector_store(repo_name: str, repo_files: List[Dict]) -> List[FileDescription]:
    file_descriptions = []
    for file in repo_files:
        description_file = llm_query(
            countSelfLoop=2,
            prompt=file['content'],
            model_role="file_description",
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
    milvus_db.insert_data(
        collection_name=GITHUB_DESCRIPTION_STRUCTURE_COLLECTION,
        data=file_descriptions,
    )
    return file_descriptions


def insert_file_requirements_to_vector_store(repo_name: str, repo_files: List[Dict]):
    file_requirements = []
    for file in repo_files:
        import_list = get_import_list(file['content'])
        file_requirements.append({
            "path": file['path'],
            "repo_name": repo_name,
            "depend_on": import_list,
        })
    file_requirements = process_raw_hierarchy(file_requirements)
    milvus_db.insert_data(
        collection_name=GITHUB_HIERARCHY_STRUCTURE_COLLECTION,
        data=file_requirements,  # xu li them
    )


def insert_idea_to_vector_store(repo_name: str, file_descriptions: List[FileDescription]):
    file_descriptions_string = FileDescriptions(
        files=file_descriptions,
    ).model_dump_json(exclude_none=True)
    idea_summary = llm_query(
        countSelfLoop=2,
        model_role="idea_summary",
        prompt=file_descriptions_string,
        model_name=app_config.LLAMA_MODEL_NAME,
    )
    embedding_idea = embedding_text(idea_summary)
    milvus_db.insert_data(
        collection_name=GITHUB_IDEA_COLLECTION,
        data=[{
            DEFAULT_TEXT_FIELD: idea_summary,
            DEFAULT_EMBEDDING_FIELD: embedding_idea,
            # Metadata
            "repo_name": repo_name,
        }],
    )