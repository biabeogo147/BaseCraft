from tqdm import tqdm
from typing import List
from pymilvus.milvus_client import IndexParams
from app.utils.utils import embedding_text
from pymilvus import MilvusClient, DataType, CollectionSchema
from app.config.app_config import IS_METADATA, MILVUS_USER, MILVUS_PASSWORD, MILVUS_HOST, \
    EMBED_VECTOR_DIM, RENEW_DB, KNOWLEDGE_BASE_DB, GITHUB_RAW_CODE_COLLECTION, RENEW_COLLECTIONS, \
    INSERT_RANDOM_DATA, DEFAULT_EMBEDDING_FIELD, DEFAULT_TEXT_FIELD, DEFAULT_METRIC_TYPE, \
    INIT_COLLECTIONS

client = MilvusClient(
    uri=MILVUS_HOST,
    token=f"{MILVUS_USER}:{MILVUS_PASSWORD}",
)
# client = MilvusClient("./milvus_demo.db")


def setup_vector_store():
    if RENEW_DB:
        drop_db(KNOWLEDGE_BASE_DB)
    init_db(KNOWLEDGE_BASE_DB)
    if INSERT_RANDOM_DATA:
        insert_random_data(KNOWLEDGE_BASE_DB, GITHUB_RAW_CODE_COLLECTION)


def init_db(db_name: str):
    """Initialize the database and index with LlamaIndex."""
    try:
        databases = client.list_databases()
        if db_name not in databases:
            print(f"Initializing database {db_name}...")
            create_db(db_name)
            for collection_name in INIT_COLLECTIONS:
                create_collection(collection_name)
        else:
            client.use_database(db_name=db_name)
            print(f"Using existing database {db_name}.")
            for collection_name in RENEW_COLLECTIONS:
                drop_collection(collection_name)
                create_collection(collection_name)
    except Exception as e:
        print(f"Failed to initialize database: {e}")
        raise


def drop_db(db_name: str):
    """Drop the database and all its collections."""
    try:
        if db_name not in client.list_databases():
            print(f"Database {db_name} does not exist.")
            return
        client.use_database(db_name=db_name)
        collections = client.list_collections()
        for collection in collections:
            client.drop_collection(collection_name=collection)
            print(f"Collection {collection} dropped.")
        client.drop_database(db_name=db_name)
        print(f"Database {db_name} dropped.")
    except Exception as e:
        print(f"Failed to drop database: {e}")
        raise


def create_db(db_name: str):
    """database.replica.number (integer): The number of replicas for the specified database.
    database.resource.groups (string): The names of the resource groups associated with the specified database in a comma-separated list.
    database.diskquota.mb (integer): The maximum size of the disk space for the specified database, in megabytes (MB).
    database.max.collections (integer): The maximum number of collections allowed in the specified database.
    database.force.deny.writing (boolean): Whether to force the specified database to deny writing operations.
    database.force.deny.reading (boolean): Whether to force the specified database to deny reading operations."""
    client.create_database(
        db_name=db_name,
        properties=None,
    )
    client.use_database(
        db_name=db_name,
    )
    print(f"Database {db_name} created and set as current database.")


def create_schema() -> CollectionSchema:
    schema = MilvusClient.create_schema(
        auto_id=True,
        enable_dynamic_field=IS_METADATA,
    )
    schema.add_field(
        datatype=DataType.INT64,
        element_type=None,
        field_name="id",
        is_primary=True,
        auto_id=False,
        dim=None,
    )
    schema.add_field(
        field_name=DEFAULT_EMBEDDING_FIELD,
        datatype=DataType.FLOAT_VECTOR,
        dim=EMBED_VECTOR_DIM,
        element_type=None,
        is_primary=False,
        auto_id=False,
    )
    schema.add_field(
        field_name=DEFAULT_TEXT_FIELD,
        datatype=DataType.VARCHAR,
        element_type=None,
        is_primary=False,
        max_length=500,
        auto_id=False,
    )
    return schema


def create_index_params() -> IndexParams:
    index_params = client.prepare_index_params()
    index_params.add_index(
        field_name="id",
        index_type="AUTOINDEX",
    )
    index_params.add_index(
        index_type="AUTOINDEX",
        index_name="dense_vector_index",
        metric_type=DEFAULT_METRIC_TYPE,
        field_name=DEFAULT_EMBEDDING_FIELD,
    )
    return index_params


def create_collection(collection_name: str):
    schema = create_schema()
    index_params = create_index_params()
    client.create_collection(
        collection_name=collection_name,
        index_params=index_params,
        schema=schema,
    )
    print(f"Collection {collection_name} created.")


def drop_collection(collection_name: str):
    try:
        if collection_name not in client.list_collections():
            print(f"Collection {collection_name} does not exist.")
            return
        client.drop_collection(collection_name=collection_name)
        print(f"Collection {collection_name} dropped.")
    except Exception as e:
        print(f"Failed to drop collection: {e}")
        raise


def insert_random_data(db_name: str, collection_name: str):
    client.use_database(
        db_name=db_name,
    )

    data = []
    text_lines = [
        "There are 2 people in the kitchen.",
        "There are 5 people in the bathroom.",
        "There are 10 people in the living room.",
        "Van Nhan is a Dau Buoi."
    ]
    for i, line in enumerate(tqdm(text_lines, desc="Creating embeddings")):
        data.append({
            "dense_vector": embedding_text(line),
            "content": line,
        })
    client.insert(collection_name=collection_name, data=data)
    print(f"Inserted {len(data)} data into collection {collection_name}.")


def insert_data(collection_name: str, data: List[dict]):
    try:
        client.insert(collection_name=collection_name, data=data)
        print(f"Inserted {len(data)} data into collection {collection_name}.")
    except Exception as e:
        print(f"Failed to insert data: {e}")
        raise