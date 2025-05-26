from google import genai
from app.config import app_config
from app.utils.utils import llm_query
from llama_index.llms.google_genai import GoogleGenAI
from llama_index.core.base.llms.types import ChatMessage
from app.config.llama_index_config import get_llama_index_model
from app.config.app_config import GOOGLE_API_KEY, GEMINI_3n_e4b_MODEL_NAME


# print(GOOGLE_API_KEY)

# Gemini genai
# client = genai.Client(api_key=GOOGLE_API_KEY)
# response = client.models.generate_content(
#     model=GEMINI_3n_e4b_MODEL_NAME, contents="Explain how AI works in a few words"
# )
# print(response.text)

# Llama index
# llm = get_llama_index_model(GEMINI_3n_e4b_MODEL_NAME)
# response_llama_index = llm.chat(
#     [ChatMessage(role="user", content="Explain how AI works in a few words"),
#      ChatMessage(role="system", content="You are a helpful assistant.")],
# )
# response_complete = llm.complete("Explain how AI works in a few words")
# print(response_llama_index.message.content)
# print(response_complete)

# Structured response
idea_result = llm_query(
    prompt="Generate a simple Flappy bird game using python.",
    count_self_loop=2,
    model_role="idea",
    # context=rag_query,
    model_name=app_config.MODEL_USING,
)
print(idea_result)