from typing import List
from llama_index.core import Document
from langchain_text_splitters import Language
from langchain.text_splitter import RecursiveCharacterTextSplitter


def split_text(text: str) -> List:
    """
    Splits the input text into chunks of specified size and overlap.

    Args:
        text (str): The text to be split.

    Returns:
        list: A list of text chunks.
    """
    try:
        text_splitter = RecursiveCharacterTextSplitter.from_language(
            chunk_size=1000,
            chunk_overlap=0,
            language=Language.PYTHON,
        )
        chunks = text_splitter.split_text(text)
        return chunks
    except Exception as e:
        print(f"Error splitting text: {e}")
        return []

if __name__ == "__main__":
    text = Document.example().text
    for chunk in split_text(text):
        print(chunk)
        print("-----")
        print("-----")