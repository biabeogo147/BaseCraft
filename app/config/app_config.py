import os

IS_OLLAMA = True
EMBED_VECTOR_DIM = 1024
VECTORDB_NAME = "milvus"

QWEN_MODEL_NAME = "qwen2.5-coder:14b"
LLAMA_MODEL_NAME = "llama3.2:1b"
MXBAI_EMBED_LARGE_MODEL_NAME = "mxbai-embed-large"

OLLAMA_HOST = "http://localhost:11434"
VLLM_HOST = "http://localhost:11435"

MILVUS_HOST = "http://localhost:19530"
MILVUS_USER = "root"
MILVUS_PASSWORD = "Milvus"
GITHUB_DB = "github_knowledge_base"
RAG_GITHUB_COLLECTION = "rag_github_collection"
INSERT_RANDOM_DATA = False
RENEW_COLLECTION = False
RENEW_DB = False

LLAMA_INDEX_DB = "default"
LLAMA_INDEX_COLLECTION = "llama_index_collection"

GITHUB_API_KEY = os.getenv("GITHUB_API_KEY")
REPO_NAMES = [
    "biabeogo147/Magic-wand",
]
FILE_TYPE_MAPPING = {
    '.py': 'code', '.txt': 'code', '.md': 'code', '.json': 'code',
    '.yml': 'code', '.yaml': 'code', '.html': 'code', '.css': 'code',
    '.js': 'code', '.java': 'code', '.c': 'code', '.cpp': 'code',
    '.h': 'code', '.sh': 'code', '.xml': 'code',

    '.png': 'image', '.jpg': 'image', '.jpeg': 'image', '.gif': 'image',
    '.bmp': 'image', '.svg': 'image', '.ico': 'image',

    '.zip': 'archive', '.tar': 'archive', '.gz': 'archive', '.rar': 'archive',

    '.exe': 'binary', '.bin': 'binary', '.dll': 'binary',

    '.pdf': 'document', '.doc': 'document', '.docx': 'document',
}