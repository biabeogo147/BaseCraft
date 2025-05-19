import json
from typing import Tuple
from app.config import app_config
from app.model.model_query.base_ollama_query import query_ollama
from app.model.model_output.pydantic.hierarchy_structure_schema import DirectoryHierarchy
from app.model.model_output.pydantic.description_structure_schema import DirectoryDescription
from app.model.model_output.pydantic.combine_hierarchy_and_description_schema import FileCombined, DirectoryCombined


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


def save(result: str, output_file: str) -> None:
    with open(output_file, "w", encoding="utf-8") as f:
        f.write(result)
    print(f"Response saved to {output_file}")


def query_and_save(prompt, model_role, model_name, output_file):
    print(f"Generating {model_role}...")
    result = query_ollama(prompt, model_role, model_name)
    save(result, output_file)
    return result


def generate_script(prompt: str, root_dir: str) -> Tuple[str, str, str]:
    idea_result = query_and_save(
        prompt=prompt,
        model_role="idea",
        model_name=app_config.LLAMA_MODEL_NAME,
        output_file=f"{root_dir}\\idea_model_response.json"
    )
    structure_result = query_and_save(
        prompt=idea_result,
        model_role="structure",
        model_name=app_config.LLAMA_MODEL_NAME,
        output_file=f"{root_dir}\\structure_model_response.json"
    )
    hierarchy_result = query_and_save(
        prompt=structure_result,
        model_role="hierarchy_structure",
        model_name=app_config.LLAMA_MODEL_NAME,
        output_file=f"{root_dir}\\hierarchy_model_response.json"
    )

    combine_result = combine_results(structure_result, hierarchy_result)
    save(combine_result, f"{root_dir}\\combined_model_response.json")

    programming_result = query_and_save(
        prompt=combine_result,
        model_role="programming",
        model_name=app_config.LLAMA_MODEL_NAME,
        output_file=f"{root_dir}\\programming_model_response.json")

    return idea_result, structure_result, programming_result