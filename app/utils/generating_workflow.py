import os
import json
from app.config import app_config
from app.config.app_config import GITHUB_COLLECTION
from app.model.model_output.programming_schema import File
from app.vector_store.milvus.milvus_rag import query_milvus
from app.model.model_query.base_ollama_query import base_query_ollama
from app.utils.process_data_util import to_directory_order, combine_results, process_depend_on, process_file_path, save


def generate_scripts(prompt: str, root_json_files: str):
    print(f"Retrieving data from Milvus...")
    rag_query = query_milvus(prompt, GITHUB_COLLECTION)

    print("Generating idea...")
    idea_result = base_query_ollama(
        prompt=prompt,
        countSelfLoop=2,
        model_role="idea",
        context=rag_query,
        model_name=app_config.LLAMA_MODEL_NAME,
    )
    save(idea_result, f"{root_json_files}\\idea_model_response.json")

    print(f"Generating description structure...")
    description_structure_result = base_query_ollama(
        countSelfLoop=5,
        context=rag_query,
        prompt=idea_result,
        model_role="description_structure",
        model_name=app_config.LLAMA_MODEL_NAME,
    )
    description_structure_result = process_file_path(description_structure_result)
    save(description_structure_result, f"{root_json_files}\\description_structure_model_response.json")

    print(f"Generating hierarchy structure...")
    hierarchy_structure_result = base_query_ollama(
        countSelfLoop=5,
        context=rag_query,
        model_role="hierarchy_structure",
        prompt=description_structure_result,
        model_name=app_config.LLAMA_MODEL_NAME,
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
    print(f"Generating source codes...")
    root_programming_json = os.path.join(root_json_files, "programming_model_response")
    os.makedirs(root_programming_json)
    cnt = 0
    for file in directory_order.files:
        programming_result = base_query_ollama(
            prompt=file.model_dump_json(exclude_none=True),
            model_name=app_config.LLAMA_MODEL_NAME,
            model_role="programming",
            context=rag_query,
            countSelfLoop=5,
        )
        save(programming_result, f"{root_programming_json}\\{os.path.basename(file.path)}.json")
        cnt += 1
    print(f"Generated {cnt} files.")


def generate_directories_and_files(root_json_files: str, root_dir: str) -> None:
    print("Generating directories and files...")

    for file in os.listdir(f"{root_json_files}\\programming_model_response"):
        with open(f"{root_json_files}\\programming_model_response\\{file}", "r", encoding="utf-8") as f:
            programming_result = File(**json.loads(f.read()))

        if programming_result:
            file_path = os.path.join(root_dir, programming_result.path)
            os.makedirs(os.path.dirname(file_path), exist_ok=True)
            with open(file_path, "w", encoding="utf-8") as f:
                f.write(programming_result.content)

    print("Directories and files generated successfully.")