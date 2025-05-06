from tqdm import tqdm
from typing import List
from app.config import default_config
from pymilvus.milvus_client import IndexParams
from pymilvus import MilvusClient, DataType, CollectionSchema
from app.model.model_query.base_ollama_query import embedding_ollama

# client = MilvusClient(
#     uri=default_config.MILVUS_HOST,
#     token=f"{default_config.MILVUS_USER}:{default_config.MILVUS_PASSWORD}",
# )
client = MilvusClient("./milvus_demo.db")


def init_db():
    databases = client.list_databases()
    if not default_config.GITHUB_DB in databases:
        create_github_db()
        schema = create_github_schema()
        index_params = create_github_index_params()
        create_github_rag_collection(schema=schema, index_params=index_params)
        insert_data()


def create_github_db():
    """database.replica.number (integer): The number of replicas for the specified database.
    database.resource.groups (string): The names of the resource groups associated with the specified database in a comma-separated list.
    database.diskquota.mb (integer): The maximum size of the disk space for the specified database, in megabytes (MB).
    database.max.collections (integer): The maximum number of collections allowed in the specified database.
    database.force.deny.writing (boolean): Whether to force the specified database to deny writing operations.
    database.force.deny.reading (boolean): Whether to force the specified database to deny reading operations."""
    client.create_database(
        db_name=default_config.GITHUB_DB,
        properties=None,
    )
    client.use_database(
        db_name=default_config.GITHUB_DB,
    )


def create_github_schema() -> CollectionSchema:
    schema = MilvusClient.create_schema(
        enable_dynamic_field=True,
    )
    schema.add_field(
        datatype=DataType.INT64,
        field_name="id",
        is_primary=True,
        auto_id=True,
        element_type=None,
        dim=None,
    )
    schema.add_field(
        dim=default_config.EMBED_VECTOR_DIM,
        datatype=DataType.FLOAT_VECTOR,
        field_name="dense_vector",
        element_type=None,
        is_primary=False,
        auto_id=False,
    )
    schema.add_field(
        datatype=DataType.STRING,
        field_name="content",
        element_type=None,
        is_primary=False,
        auto_id=False,
        dim=None,
    )
    return schema


def create_github_index_params() -> IndexParams:
    index_params = client.prepare_index_params()
    index_params.add_index(
        field_name="id",
        index_type="AUTOINDEX",
    )
    index_params.add_index(
        index_name="dense_vector_index",
        field_name="dense_vector",
        index_type="AUTOINDEX",
        metric_type="COSINE"
    )
    return index_params


def create_github_rag_collection(schema: CollectionSchema, index_params: IndexParams):
    client.create_collection(
        collection_name=default_config.RAG_GITHUB_COLLECTION,
        index_params=index_params,
        schema=schema,
    )


def emb_text(line) -> List[float]:
    if default_config.IS_OLLAMA:
        result = embedding_ollama([line], model_name=default_config.MXBAI_EMBED_LARGE_MODEL_NAME)
        return result[0]
    else:
        return [0] * default_config.EMBED_VECTOR_DIM


def insert_data():
    data = []
    text_lines = [
        "This is a test line 1.",
        "This is a test line 2.",
        "This is a test line 3.",
    ]
    for i, line in enumerate(tqdm(text_lines, desc="Creating embeddings")):
        data.append({"dense_vector": emb_text(line), "content": line})
    client.insert(collection_name=default_config.RAG_GITHUB_COLLECTION, data=data)