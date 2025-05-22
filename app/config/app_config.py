import os

IS_OLLAMA = True
CACHE_NAME = "redis"
EMBED_VECTOR_DIM = 1024
VECTORDB_NAME = "milvus"

RENEW_DB = False
RENEW_CACHE = True
IS_METADATA = True
INSERT_RANDOM_DATA = False
DEFAULT_TEXT_FIELD = "text"
DEFAULT_METRIC_TYPE = "COSINE"
KNOWLEDGE_BASE_DB = "knowledge_base"
DEFAULT_EMBEDDING_FIELD = "embedding"

USER_IDEA_COLLECTION = "user_idea"
USER_RAW_CODE_COLLECTION = "user_raw_code"
USER_HIERARCHY_STRUCTURE_COLLECTION = "user_hierarchy_structure"
USER_DESCRIPTION_STRUCTURE_COLLECTION = "user_description_structure"

GITHUB_IDEA_COLLECTION = "github_idea"
GITHUB_RAW_CODE_COLLECTION = "github_raw_code"
GITHUB_HIERARCHY_STRUCTURE_COLLECTION = "github_hierarchy_structure"
GITHUB_DESCRIPTION_STRUCTURE_COLLECTION = "github_description_structure"

# List collection names to renew
RENEW_COLLECTIONS = [
    GITHUB_IDEA_COLLECTION, GITHUB_RAW_CODE_COLLECTION, GITHUB_HIERARCHY_STRUCTURE_COLLECTION, GITHUB_DESCRIPTION_STRUCTURE_COLLECTION,
    USER_IDEA_COLLECTION, USER_RAW_CODE_COLLECTION,  USER_HIERARCHY_STRUCTURE_COLLECTION, USER_DESCRIPTION_STRUCTURE_COLLECTION,
]

# List collection names to initialize
INIT_COLLECTIONS = [
    GITHUB_IDEA_COLLECTION, GITHUB_RAW_CODE_COLLECTION, GITHUB_HIERARCHY_STRUCTURE_COLLECTION, GITHUB_DESCRIPTION_STRUCTURE_COLLECTION,
    USER_IDEA_COLLECTION, USER_RAW_CODE_COLLECTION,  USER_HIERARCHY_STRUCTURE_COLLECTION, USER_DESCRIPTION_STRUCTURE_COLLECTION,
]

LLAMA_MODEL_NAME = "llama3.2:3b"
QWEN_MODEL_NAME = "qwen2.5-coder:14b"
MXBAI_EMBED_LARGE_MODEL_NAME = "mxbai-embed-large"

OLLAMA_HOST = "http://localhost:11434"
VLLM_HOST = "http://localhost:11435"

MILVUS_HOST = "http://localhost:19530"
MILVUS_USER = "root"
MILVUS_PASSWORD = "Milvus"

REDIS_HOST = "localhost"
REDIS_PORT = 6379
REDIS_GITHUB_DB = 0
REDIS_USER_PROJECT_DB = 1
REDIS_PASSWORD = "PASSWORD"

GITHUB_API_KEY = os.getenv("GITHUB_API_KEY")
REPO_NAMES = [
    "sourabhv/FlapPyBird",
    "clear-code-projects/FlappyBird_Python",
    "LeonMarqs/Flappy-bird-python",
    "undercase/FlappyKivy",
    "mohitk05/FlappyBird",
    "techwithtim/Flappy-Bird",
    "filhoweuler/python-flappybird",
    "tjwei/Flappy-Turtle",

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