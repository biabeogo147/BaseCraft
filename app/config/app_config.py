import os

IS_OLLAMA = True
CACHE_NAME = "redis"
EMBED_VECTOR_DIM = 1024
VECTORDB_NAME = "milvus"

RENEW_DB = False
RENEW_CACHE = True
IS_METADATA = True
RENEW_COLLECTION = True
INSERT_RANDOM_DATA = False
DEFAULT_TEXT_FIELD = "text"
DEFAULT_METRIC_TYPE = "COSINE"
GITHUB_DB = "github_knowledge_base"
DEFAULT_EMBEDDING_FIELD = "embedding"
RAG_GITHUB_COLLECTION = "rag_github_collection"

QWEN_MODEL_NAME = "qwen2.5-coder:14b"
LLAMA_MODEL_NAME = "llama3.2:1b"
MXBAI_EMBED_LARGE_MODEL_NAME = "mxbai-embed-large"

OLLAMA_HOST = "http://localhost:11434"
VLLM_HOST = "http://localhost:11435"

MILVUS_HOST = "http://localhost:19530"
MILVUS_USER = "root"
MILVUS_PASSWORD = "Milvus"

REDIS_HOST = "localhost"
REDIS_PORT = 6379
REDIS_GITHUB_RAG_DB = 0
REDIS_PASSWORD = "PASSWORD"

GITHUB_API_KEY = os.getenv("GITHUB_API_KEY")
REPO_NAMES = [
    "biabeogo147/Magic-wand",
    "CorentinJ/Real-Time-Voice-Cloning",
    "feder-cr/Jobs_Applier_AI_Agent_AIHawk",
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
EXTENSION_TO_LANGUAGE = {
    '.py': 'Python',
    '.java': 'Java',
    '.js': 'JavaScript',
    '.cpp': 'C++',
    '.cc': 'C++',
    '.cxx': 'C++',
    '.hpp': 'C++',
    '.h': 'C/C++',
    '.c': 'C',
    '.cs': 'C#',
    '.rb': 'Ruby',
    '.php': 'PHP',
    '.go': 'Go',
    '.rs': 'Rust',
    '.swift': 'Swift',
    '.kt': 'Kotlin',
    '.ts': 'TypeScript',
    '.tsx': 'TypeScript',
    '.html': 'HTML',
    '.css': 'CSS',
    '.sh': 'Shell Script',
    '.sql': 'SQL',
    '.md': 'Markdown',
    '.json': 'JSON',
    '.xml': 'XML',
    '.yaml': 'YAML',
    '.yml': 'YAML',
}