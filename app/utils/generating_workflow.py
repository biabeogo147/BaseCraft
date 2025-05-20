import os
import json
from typing import Tuple, List
from app.config import app_config
from app.model.model_output.programming_schema import File
from app.model.model_query.base_ollama_query import base_query_ollama
from app.model.model_output.hierarchy_structure_schema import DirectoryHierarchy
from app.model.model_output.description_structure_schema import DirectoryDescription
from app.model.model_output.combine_hierarchy_and_description_schema import FileCombined, DirectoryCombined, \
    DirectoryOrder, FileOrder


def get_edge(directoryCombined: DirectoryCombined) -> Tuple[List[List[int]], List[int], int]:
    edges = []
    files = directoryCombined.files
    num_node = len(files)
    count_in = [0] * num_node
    for i, file in enumerate(files):
        for j, depend_file in file.depend_on:
            edges[j].append(i)
    return edges, count_in, num_node


def topological_sort(edges: List[List[int]], count_in: List[int], num_node: int) -> List[Tuple[int, int]]:
    order = []
    for i in range(num_node):
        if count_in[i] == 0:
            order.append((i, 0))

    cnt = 0
    while cnt < len(order):
        node, node_order = order[cnt]
        for neighbor in edges[node]:
            count_in[neighbor] -= 1
            if count_in[neighbor] == 0:
                order.append((neighbor, node_order + 1))

    return order


def to_directory_order(directoryCombined: DirectoryCombined) -> DirectoryOrder:
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
    directory_order = DirectoryOrder(
        directories=directoryCombined.directories,
        files=[file for file in file_order]
    )
    return directory_order


def combine_results(structure_result: str, hierarchy_result: str) -> DirectoryCombined:
    json_structure = json.loads(structure_result)
    json_hierarchy = json.loads(hierarchy_result)
    hierarchy = DirectoryHierarchy(**json_hierarchy)
    structure = DirectoryDescription(**json_structure)
    combined_dirs = list(set(structure.directories + hierarchy.directories))

    combined_files = []
    hierarchy_files = {f.path: f.depend_on for f in hierarchy.files}
    structure_files = {f.path: f.description for f in structure.files}
    all_paths = set(structure_files.keys()) | set(hierarchy_files.keys())

    for path in all_paths:
        file_combined = FileCombined(
            path=path,
            description=structure_files.get(path, ""),
            depend_on=hierarchy_files.get(path, [])
        )
        combined_files.append(file_combined)

    combined = DirectoryCombined(
        directories=combined_dirs,
        files=combined_files
    )

    return combined


def process_directories(structure_result: str) -> str:
    json_structure = json.loads(structure_result)
    structure = DirectoryDescription(**json_structure)

    new_directories = []
    directories = structure.directories
    for directory in directories:
        if os.path.isdir(directory):
            new_directories.append(directory)
    structure.directories = new_directories

    return structure.model_dump_json(exclude_none=True, indent=4)


def save(result: str, output_file: str) -> None:
    with open(output_file, "w", encoding="utf-8") as f:
        f.write(result)
    print(f"Response saved to {output_file}")


def generate_script(prompt: str, root_json_files: str):
    print("Generating idea...")
    idea_result = base_query_ollama(
        prompt=prompt,
        model_role="idea",
        model_name=app_config.LLAMA_MODEL_NAME,
    )
    save(idea_result, f"{root_json_files}\\idea_model_response.json")

    print(f"Generating description structure...")
    description_structure_result = base_query_ollama(
        prompt=idea_result,
        model_role="description_structure",
        model_name=app_config.LLAMA_MODEL_NAME,
    )
    description_structure_result = process_directories(description_structure_result)
    save(description_structure_result, f"{root_json_files}\\description_structure_model_response.json")

    print(f"Generating hierarchy structure...")
    hierarchy_structure_result = base_query_ollama(
        prompt=description_structure_result,
        model_role="hierarchy_structure",
        model_name=app_config.LLAMA_MODEL_NAME,
    )
    save(hierarchy_structure_result, f"{root_json_files}\\hierarchy_structure_model_response.json")

    print(f"Generating combined structure...")
    combine_result = combine_results(description_structure_result, hierarchy_structure_result)
    save(combine_result.model_dump_json(exclude_none=True, indent=4), f"{root_json_files}\\combined_model_response.json")

    print(f"Generating order...")
    directory_order = to_directory_order(combine_result)
    save(directory_order.model_dump_json(exclude_none=True, indent=4), f"{root_json_files}\\directory_order.json")

    # Use async query for prompts have equal order
    print(f"Generating source codes...")
    cnt = 0
    for file in directory_order.files:
        programming_result = base_query_ollama(
            prompt=file.model_dump_json(exclude_none=True),
            model_name=app_config.LLAMA_MODEL_NAME,
            model_role="programming",
        )
        save(programming_result, f"{root_json_files}\\programming_model_response\\{os.path.basename(file.path)}.json")
        cnt += 1
    print(f"Generated {cnt} files.")


def generate_directories_and_files(root_json_files: str, root_dir: str) -> None:
    print("Generating directories and files...")

    with open(f"{root_json_files}\\description_structure_response.json", "w", encoding="utf-8") as f:
        description_structure_result = DirectoryDescription(**json.loads(f.read()))

    if description_structure_result:
        for directory in description_structure_result.directories:
            os.makedirs(os.path.join(root_dir, directory), exist_ok=True)

        for file in os.listdir(f"{root_json_files}\\programming_model_response"):
            programming_result = None
            with open(f"{root_json_files}\\programming_model_response\\{file}", "r", encoding="utf-8") as f:
                programming_result = File(**json.loads(f.read()))

            if programming_result:
                file_path = os.path.join(root_dir, programming_result.path)
                os.makedirs(os.path.dirname(file_path), exist_ok=True)
                with open(file_path, "w", encoding="utf-8") as f:
                    f.write(programming_result.content)

    print("Directories and files generated successfully.")