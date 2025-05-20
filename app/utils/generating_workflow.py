import os
import json
from typing import Tuple, List
from app.config import app_config
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


def topological_sort(edges: List[List[int]], count_in: List[int], num_node: int) -> List[int]:
    order = []
    for i in range(num_node):
        if count_in[i] == 0:
            order.append(i)

    cnt = 0
    while cnt < len(order):
        node = order[cnt]
        for neighbor in edges[node]:
            count_in[neighbor] -= 1
            if count_in[neighbor] == 0:
                order.append(neighbor)

    return order


def to_directory_order(directoryCombined: DirectoryCombined) -> str:
    edges, count_in, num_node = get_edge(directoryCombined)
    topo_order = topological_sort(edges, count_in, num_node)
    file_order = [
        FileOrder(
            order=order,
            path=directoryCombined.files[i].path,
            depend_on=directoryCombined.files[i].depend_on,
            description=directoryCombined.files[i].description,
        )
        for i, order in enumerate(topo_order)
    ]
    directory_order = DirectoryOrder(
        directories=directoryCombined.directories,
        files=[file for file in file_order]
    )
    return directory_order.model_dump_json(exclude_none=True, indent=4)


def combine_results(structure_result: str, hierarchy_result: str) -> str:
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

    return combined.model_dump_json(exclude_none=True, indent=4)


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


def generate_script(prompt: str, root_dir: str) -> Tuple[str, str, str]:
    idea_result = base_query_ollama(
        prompt=prompt,
        model_role="idea",
        model_name=app_config.LLAMA_MODEL_NAME,
    )
    save(idea_result, f"{root_dir}\\idea_model_response.json")

    structure_result = base_query_ollama(
        prompt=idea_result,
        model_role="structure",
        model_name=app_config.LLAMA_MODEL_NAME,
    )
    structure_result = process_directories(structure_result)
    save(structure_result, f"{root_dir}\\structure_model_response.json")

    hierarchy_result = base_query_ollama(
        prompt=structure_result,
        model_role="hierarchy_structure",
        model_name=app_config.LLAMA_MODEL_NAME,
    )
    save(hierarchy_result, f"{root_dir}\\hierarchy_model_response.json")

    combine_result = combine_results(structure_result, hierarchy_result)
    save(combine_result, f"{root_dir}\\combined_model_response.json")

    directory_order = to_directory_order(json.loads(combine_result))
    save(directory_order, f"{root_dir}\\directory_order.json")

    programming_result = base_query_ollama(
        prompt=combine_result,
        model_role="programming",
        model_name=app_config.LLAMA_MODEL_NAME,
    )
    save(programming_result, f"{root_dir}\\programming_model_response.json")

    return idea_result, structure_result, programming_result