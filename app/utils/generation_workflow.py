import os
import json
from typing import Tuple, List, Dict, Any
from app.config import app_config
from app.llm.llm_output.programming_schema import File
from app.utils.utils import save, is_file, llm_query, get_metadata
from app.llm.llm_output.hierarchy_structure_schema import FileRequirements
from app.llm.llm_output.description_structure_schema import FileDescriptions
from app.vector_store.milvus.milvus_rag import query_milvus_with_prompt, query_milvus_with_metadata
from app.config.app_config import GITHUB_IDEA_COLLECTION, GITHUB_DESCRIPTION_STRUCTURE_COLLECTION, \
    GITHUB_HIERARCHY_STRUCTURE_COLLECTION, FILE_TYPE_MAPPING, DEFAULT_TEXT_FIELD, METADATA_IDEA_COLLECTION, \
    METADATA_DESCRIPTION_STRUCTURE_COLLECTION, METADATA_HIERARCHY_STRUCTURE_COLLECTION
from app.llm.llm_output.combine_hierarchy_and_description_schema import FileCombines, FileOrders, FileOrder, \
    FileCombined


def get_depend_on_script(depend_on: List[str], code_save_dir: str) -> str:
    depend_on_script = ""
    for file in depend_on:
        if FILE_TYPE_MAPPING.get(os.path.splitext(file)[1]) == "code":
            with open(f"{code_save_dir}\\{os.path.basename(file)}_fixing.json", "r", encoding="utf-8") as f:
                depend_on_script += f"{file}:\n{f.read()}\n\n"
        else:
            depend_on_script = f"{file}\n\n"
    return depend_on_script


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


def get_hierarchy_context(idea_metadata: Dict[str, Any]) -> str:
    repo_name = idea_metadata["repo_name"]
    hierarchy_context = query_milvus_with_metadata(
        metadata={"repo_name": repo_name},
        collection_name=GITHUB_HIERARCHY_STRUCTURE_COLLECTION,
        output_fields=METADATA_HIERARCHY_STRUCTURE_COLLECTION,
    )
    hierarchy_context = sorted(hierarchy_context, key=lambda x: x["chunk_id"])
    hierarchy_context = " ".join([item[DEFAULT_TEXT_FIELD] for item in hierarchy_context])
    return hierarchy_context


def get_description_context(idea_metadata: Dict[str, Any]) -> str:
    repo_name = idea_metadata["repo_name"]
    description_context = query_milvus_with_metadata(
        metadata={"repo_name": repo_name},
        collection_name=GITHUB_DESCRIPTION_STRUCTURE_COLLECTION,
        output_fields=METADATA_DESCRIPTION_STRUCTURE_COLLECTION,
    )
    description_context = sorted(description_context, key=lambda x: x["chunk_id"])
    description_context = " ".join([item[DEFAULT_TEXT_FIELD] for item in description_context])
    return description_context


def get_idea_context(prompt: str) -> Tuple[str, List[Dict[str, Any]]]:
    idea_rag_query = query_milvus_with_prompt(
        limit=1,
        prompt=prompt,
        collection_name=GITHUB_IDEA_COLLECTION,
        output_fields=METADATA_IDEA_COLLECTION,
    )
    idea_metadata = get_metadata(
        datas=idea_rag_query,
        metadata_fields=METADATA_IDEA_COLLECTION,
    )
    repo_name = idea_metadata[0]["repo_name"]
    idea_context = query_milvus_with_metadata(
        metadata={"repo_name": repo_name},
        collection_name=GITHUB_IDEA_COLLECTION,
        output_fields=METADATA_IDEA_COLLECTION,
    )
    idea_context = sorted(idea_context, key=lambda x: x["chunk_id"])
    idea_context = " ".join([item[DEFAULT_TEXT_FIELD] for item in idea_context])
    return idea_context, idea_metadata


def generate_scripts(prompt: str, root_json_files: str):
    print("Generating idea...")

    idea_context, idea_metadata = get_idea_context(prompt)
    idea_result = llm_query(
        prompt=prompt,
        count_self_loop=1,
        model_role="idea",
        context=idea_context,
        model_name=app_config.MODEL_USING,
    )
    save(idea_result, f"{root_json_files}\\idea_model_response.json")

    print(f"Generating description structure...")
    description_context = get_description_context(idea_metadata[0])
    description_structure_result = llm_query(
        count_self_loop=1,
        prompt=idea_result,
        context=description_context,
        model_name=app_config.MODEL_USING,
        model_role="description_structure",
    )
    description_structure_result = process_file_path(description_structure_result)
    save(description_structure_result, f"{root_json_files}\\description_structure_model_response.json")

    # Get idea_rag_query metadata above to get all HIERARCHY_STRUCTURE belong to that repo
    print(f"Generating hierarchy structure...")
    hierarchy_context = get_hierarchy_context(idea_metadata[0])
    hierarchy_structure_result = llm_query(
        count_self_loop=1,
        context=hierarchy_context,
        model_role="hierarchy_structure",
        model_name=app_config.MODEL_USING,
        prompt=description_structure_result,
    )
    hierarchy_structure_result = process_depend_on(hierarchy_structure_result)
    save(hierarchy_structure_result, f"{root_json_files}\\hierarchy_structure_model_response.json")

    print(f"Generating combined structure...")
    combine_result = combine_results(description_structure_result, hierarchy_structure_result)
    save(combine_result.model_dump_json(exclude_none=True, indent=4), f"{root_json_files}\\combined_model_response.json")

    print(f"Generating order...")
    directory_order = to_directory_order(combine_result)
    save(directory_order.model_dump_json(exclude_none=True, indent=4), f"{root_json_files}\\directory_order.json")

    # Use async query for prompts have equal order
    # Dựa vào depend on để cung cấp context cho programming llm
    print(f"Generating source codes...")
    root_programming_json = os.path.join(root_json_files, "programming_model_response")
    root_fix_code_json = os.path.join(root_json_files, "fixing_model_response")
    os.makedirs(root_programming_json)
    os.makedirs(root_fix_code_json)
    cnt = 0
    for file in directory_order.files:
        if FILE_TYPE_MAPPING.get(os.path.splitext(file.path)[1]) == "code":
            programming_result = llm_query(
                context=get_depend_on_script(file.depend_on, root_fix_code_json),
                prompt=file.model_dump_json(exclude_none=True),
                model_name=app_config.MODEL_USING,
                model_role="programming",
                count_self_loop=1,
            )
            save(programming_result, f"{root_programming_json}\\{os.path.basename(file.path)}.json")
            fixing_result = llm_query(
                model_name=app_config.MODEL_USING,
                model_role="compile_error_fix",
                prompt=programming_result,
                context=file.description,
                count_self_loop=1,
            )
            save(fixing_result, f"{root_fix_code_json}\\{os.path.basename(file.path)}_fixing.json")
            cnt += 1
    print(f"Generated {cnt} files.")


def generate_directories_and_files(root_json_files: str, root_dir: str) -> None:
    print("Generating directories and files...")

    root_programming_json = os.path.join(root_json_files, "programming_model_response")
    root_fix_code_json = os.path.join(root_json_files, "fixing_model_response")

    for file in os.listdir(root_fix_code_json):
        with open(f"{root_fix_code_json}\\{file}", "r", encoding="utf-8") as f:
            programming_result = File(**json.loads(f.read()))

        if programming_result:
            file_path = os.path.join(root_dir, programming_result.path)
            os.makedirs(os.path.dirname(file_path), exist_ok=True)
            with open(file_path, "w", encoding="utf-8") as f:
                f.write(programming_result.content)

    print("Directories and files generated successfully.")