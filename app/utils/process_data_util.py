import os
import json
from github import Github
from typing import List, Dict, Tuple
from github.Repository import Repository
from langchain_text_splitters import Language
from llama_index.core.prompts import RichPromptTemplate
from langchain.text_splitter import RecursiveCharacterTextSplitter
from app.model.model_query.base_ollama_query import embedding_ollama
from app.model.model_output.hierarchy_structure_schema import FileRequirements
from app.model.model_output.description_structure_schema import FileDescriptions
from app.config.app_config import FILE_TYPE_MAPPING, GITHUB_API_KEY, EXTENSION_TO_LANGUAGE
from app.config.app_config import IS_OLLAMA, MXBAI_EMBED_LARGE_MODEL_NAME, EMBED_VECTOR_DIM
from app.model.model_output.combine_hierarchy_and_description_schema import FileCombines, FileOrders, FileOrder, \
    FileCombined

github = None


def prompt_template(context: str, previous_response: str, path: str) -> str:
    """
    Create a prompt template for model.
    """
    template_str = open(path, "r", encoding="utf-8").read()

    if not template_str:
        print("Prompt template is empty")
        return ""

    try:
        qa_template = RichPromptTemplate(template_str)
        prompt = qa_template.format(context_str=context, previous_response=previous_response)
    except Exception as e:
        print(f"Error adding context: {e}. \n Using default prompt.")
        prompt = template_str

    return prompt


def embedding_text(line) -> List[float]:
    """
    Embeds the input text using the specified embedding model.
    """
    if IS_OLLAMA:
        result = embedding_ollama([line], model_name=MXBAI_EMBED_LARGE_MODEL_NAME)
        return result[0]
    else:
        return [0] * EMBED_VECTOR_DIM


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


def get_edge(directoryCombined: FileCombines) -> Tuple[List[List[int]], List[int], int]:
    files = directoryCombined.files
    file2index = {file.path: i for i, file in enumerate(files)}

    num_node = len(files)
    count_in = [0] * num_node
    edges = [[] for _ in range(num_node)]
    for i, file in enumerate(files):
        for depend_file in file.depend_on:
            index = file2index.get(depend_file)
            if index:
                edges[index].append(i)
                count_in[i] += 1
    return edges, count_in, num_node


def topological_sort(edges: List[List[int]], count_in: List[int], num_node: int) -> List[Tuple[int, int]]:
    order = []
    for i in range(num_node):
        if count_in[i] == 0:
            order.append((i, 0))

    cnt = 0
    while cnt < len(order):
        node, node_order = order[cnt]
        cnt += 1
        for neighbor in edges[node]:
            count_in[neighbor] -= 1
            if count_in[neighbor] == 0:
                order.append((neighbor, node_order + 1))

    return order


def to_directory_order(directoryCombined: FileCombines) -> FileOrders:
    edges, count_in, num_node = get_edge(directoryCombined)
    topo_order = topological_sort(edges, count_in, num_node)
    file_order = [
        FileOrder(
            order=order[1],
            path=directoryCombined.files[order[0]].path,
            depend_on=directoryCombined.files[order[0]].depend_on,
            description=directoryCombined.files[order[0]].description,
        )
        for order in topo_order
    ]
    file_orders = FileOrders(files=[file for file in file_order])
    return file_orders


def combine_results(structure_result: str, hierarchy_result: str) -> FileCombines:
    json_structure = json.loads(structure_result)
    json_hierarchy = json.loads(hierarchy_result)
    hierarchy = FileRequirements(**json_hierarchy)
    description = FileDescriptions(**json_structure)

    combined_files = []
    all_paths = [f.path for f in description.files]
    hierarchy_files = {f.path: f.depend_on for f in hierarchy.files}
    description_files = {f.path: f.description for f in description.files}

    for path in all_paths:
        file_combined = FileCombined(
            path=path,
            depend_on=hierarchy_files.get(path, []),
            description=description_files.get(path, ""),
        )
        combined_files.append(file_combined)

    combined = FileCombines(files=combined_files)

    return combined


def is_file(path) -> bool:
    normalized_path = os.path.normpath(path)
    name = os.path.basename(normalized_path)
    _, extension = os.path.splitext(name)
    if extension:
        return True
    return False


def process_depend_on(hierarchy_result: str) -> str:
    json_hierarchy = json.loads(hierarchy_result)
    hierarchy = FileRequirements(**json_hierarchy)

    files = hierarchy.files
    for i, file in enumerate(files):
        for depend_file in file.depend_on:
            if not is_file(depend_file) and depend_file != file.path:
                files[i].depend_on.remove(depend_file)

    return hierarchy.model_dump_json(exclude_none=True, indent=4)


def process_file_path(structure_result: str) -> str:
    json_structure = json.loads(structure_result)
    structure = FileDescriptions(**json_structure)

    new_files = []
    files = structure.files
    for file in files:
        if is_file(file.path):
            new_files.append(file)
    structure.files = new_files

    return structure.model_dump_json(exclude_none=True, indent=4)


def save(result: str, output_file: str) -> None:
    with open(output_file, "w", encoding="utf-8") as f:
        f.write(result)
    print(f"Response saved to {output_file}")


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
