import os
import re
from github import Github
from typing import List, Dict
from app.config import app_config
from github.Repository import Repository
from app.vector_store.milvus import milvus_db
from app.utils.utils import embedding_text, llm_query
from langchain_text_splitters import Language, RecursiveCharacterTextSplitter
from app.llm.llm_output.description_structure_schema import FileDescription, FileDescriptions
from app.config.app_config import LANGUAGE_LANGCHAIN, DEFAULT_TEXT_FIELD, DEFAULT_EMBEDDING_FIELD, \
    GITHUB_RAW_CODE_COLLECTION, GITHUB_DESCRIPTION_STRUCTURE_COLLECTION, GITHUB_HIERARCHY_STRUCTURE_COLLECTION, \
    GITHUB_IDEA_COLLECTION, FILE_TYPE_MAPPING, EXTENSION_TO_LANGUAGE, GITHUB_API_KEY, LANGUAGE_PATTERNS, MODULE_TO_PATH

_github = None


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
        GitHub: A GitHub instance.
    """
    global _github
    if _github is None:
        if GITHUB_API_KEY is None:
            raise ValueError("GITHUB_API_KEY environment variable not set")
        _github = Github(GITHUB_API_KEY)
    return _github


def extract_modules_from_line(line, language):
    """Extract module names from a line of code based on the programming language."""
    patterns = LANGUAGE_PATTERNS[language]
    modules = []

    for pattern_type, pattern in patterns.items():
        match = re.search(pattern, line.strip())
        if match:
            modules.append(match.group(1))

    return modules


def get_depend_on(project_files: List[Dict]) -> List[Dict]:
    """Analyze import statements in project files and print found modules."""
    project_file_paths = set()
    for file in project_files:
        file["path"] = os.path.normpath(file["path"])
        project_file_paths.add(file["path"])

    hierarchy_structure = []
    for file in project_files:
        new_file = {"path": os.path.normpath(file["path"])}
        language = EXTENSION_TO_LANGUAGE.get(os.path.splitext(new_file["path"])[1].lower())
        if not language in LANGUAGE_PATTERNS:
            continue

        new_file["depend_on"] = []
        file_path = new_file["path"]
        print(f"\nAnalyze hierarchy structure of file: {file_path} ({language})")
        for line in file["content"].splitlines():
            modules = extract_modules_from_line(line, language)
            for module in modules:
                module_paths = MODULE_TO_PATH[language]("", module)
                for module_path in module_paths:
                    if module_path in project_file_paths:
                        print(f"Find: '{module}' in project")
                        new_file["depend_on"].append(module_path)
                        break
        hierarchy_structure.append(new_file)
    return hierarchy_structure


def insert_raw_code_to_vector_store(repo_name: str, repo_files: List[Dict]):
    """
    Insert raw code into the vector store.
    """
    print("Inserting raw code into vector store...")

    for file in repo_files:
        raw_source_code = []
        chunks = split_source_code(file['content'], LANGUAGE_LANGCHAIN.get(file['language'], Language.PYTHON)) if file['content'] else []
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
        print(f"\nInserting raw code for: {file['path']}")
        milvus_db.insert_data(
            collection_name=GITHUB_RAW_CODE_COLLECTION,
            data=raw_source_code,
        )
        print(f"File: {file['path']} raw code inserted.\n")

    print("Raw code insertion completed.")


def insert_file_descriptions_to_vector_store(repo_name: str, repo_files: List[Dict]) -> List[Dict]:
    """Insert file descriptions into the vector store."""
    print("Inserting file descriptions into vector store...")

    file_descriptions_all = []
    for file in repo_files:
        file_descriptions = []
        description_file = llm_query(
            count_self_loop=2,
            prompt=file['content'],
            model_role="file_description",
            model_name=app_config.MODEL_USING,
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
        print("Inserting file description for:", file['path'])
        milvus_db.insert_data(
            collection_name=GITHUB_DESCRIPTION_STRUCTURE_COLLECTION,
            data=file_descriptions,
        )
        file_descriptions_all.extend(file_descriptions)
        print(f"File: {file['path']} descriptions inserted.")

    print("File descriptions insertion completed.")

    return file_descriptions_all


def insert_file_requirements_to_vector_store(repo_name: str, repo_files: List[Dict]):
    """Insert file requirements into the vector store."""
    print("Inserting file requirements into vector store...")

    hierarchy_structure = get_depend_on(repo_files)
    for hierarchy in hierarchy_structure:
        if len(hierarchy["depend_on"]):
            file_hierarchies = []
            split_depend_on = split_text(str(hierarchy["depend_on"]))
            for i, depend_on in enumerate(split_depend_on):
                embedding_depend_on = embedding_text(depend_on)
                file_hierarchies.append({
                        DEFAULT_TEXT_FIELD: depend_on,
                        DEFAULT_EMBEDDING_FIELD: embedding_depend_on,
                        # Metadata
                        "chunk_index": i,
                        "repo_name": repo_name,
                        "path": hierarchy["path"],
                    })
            milvus_db.insert_data(
                collection_name=GITHUB_HIERARCHY_STRUCTURE_COLLECTION,
                data=file_hierarchies,
            )

    print("File requirements insertion completed.")


def insert_idea_to_vector_store(repo_name: str, file_descriptions: List[Dict]):
    """Insert idea summaries into the vector store."""
    print("Inserting idea summaries into vector store...")

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
        model_name=app_config.MODEL_USING,
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

    print("Idea summaries insertion completed.")