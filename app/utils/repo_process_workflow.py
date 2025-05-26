import os
from github import Github
from typing import List, Dict
from app.config import app_config
from github.Repository import Repository
from app.vector_store.milvus import milvus_db
from app.utils.utils import is_file, embedding_text, llm_query
from langchain_text_splitters import Language, RecursiveCharacterTextSplitter
from app.llm.llm_output.description_structure_schema import FileDescription, FileDescriptions
from app.config.app_config import LANGUAGE_LANGCHAIN, DEFAULT_TEXT_FIELD, DEFAULT_EMBEDDING_FIELD, \
    GITHUB_RAW_CODE_COLLECTION, GITHUB_DESCRIPTION_STRUCTURE_COLLECTION, GITHUB_HIERARCHY_STRUCTURE_COLLECTION, \
    GITHUB_IDEA_COLLECTION, FILE_TYPE_MAPPING, EXTENSION_TO_LANGUAGE, GITHUB_API_KEY

github = None


def split_source_code(text: str, language: Language) -> List[str]:
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
            chunk_size=450,
            chunk_overlap=0,
            language=language,
        )
        chunks = text_splitter.split_text(text)
        return chunks
    except Exception as e:
        print(f"Error splitting text: {e}")
        return []


def split_text(text: str) -> List[str]:
    """
    Splits the input text into chunks of specified size and overlap.

    Args:
        text (str): The text to be split.

    Returns:
        list: A list of text chunks.
    """
    try:
        text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=450,
            chunk_overlap=0,
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


def process_raw_hierarchy(files: List[Dict]) -> List[Dict]: #Xu li them
    """
    Process the raw hierarchy data to ensure all dependencies are valid.
    """
    processed_hierarchy = []
    for file in files:
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


def insert_file_descriptions_to_vector_store(repo_name: str, repo_files: List[Dict]) -> List[Dict]:
    file_descriptions_all = []
    for file in repo_files:
        file_descriptions = []
        description_file = llm_query(
            count_self_loop=2,
            prompt=file['content'],
            model_role="file_description",
            model_name=app_config.LLAMA_MODEL_NAME,
        )
        split_description_file = split_text(description_file)
        for i, description in enumerate(split_description_file):
            embedding_description = embedding_text(description)
            file_descriptions.append({
                DEFAULT_TEXT_FIELD: description,
                DEFAULT_EMBEDDING_FIELD: embedding_description,
                # Metadata
                "chunk_index": i,
                "path": file['path'],
                "repo_name": repo_name,
            })
        milvus_db.insert_data(
            collection_name=GITHUB_DESCRIPTION_STRUCTURE_COLLECTION,
            data=file_descriptions,
        )
        file_descriptions_all.extend(file_descriptions)
    return file_descriptions_all


def insert_file_requirements_to_vector_store(repo_name: str, repo_files: List[Dict]):
    file_requirements = []
    for file in repo_files:
        if file['type'] == 'code':
            import_list = get_import_list(file['content'])
            file_requirements.append({
                "path": file['path'],
                "repo_name": repo_name,
                "depend_on_raw_path": import_list,
            })
    file_requirements = process_raw_hierarchy(file_requirements)
    milvus_db.insert_data(
        collection_name=GITHUB_HIERARCHY_STRUCTURE_COLLECTION,
        data=file_requirements,  #Xu li them
    )


def insert_idea_to_vector_store(repo_name: str, file_descriptions: List[Dict]):
    file_descriptions_string = FileDescriptions(
        files=[
            FileDescription(
                path=file['path'],
                description=file[DEFAULT_TEXT_FIELD],
            ) for file in file_descriptions
        ]
    ).model_dump_json(exclude_none=True)
    idea_summary = llm_query(
        count_self_loop=2,
        model_role="idea_summary",
        prompt=file_descriptions_string,
        model_name=app_config.LLAMA_MODEL_NAME,
    )
    split_idea_summary = split_text(idea_summary)
    for i, idea in enumerate(split_idea_summary):
        embedding_idea = embedding_text(idea)
        milvus_db.insert_data(
            collection_name=GITHUB_IDEA_COLLECTION,
            data=[{
                DEFAULT_TEXT_FIELD: idea,
                DEFAULT_EMBEDDING_FIELD: embedding_idea,
                # Metadata
                "chunk_index": i,
                "repo_name": repo_name,
            }],
        )