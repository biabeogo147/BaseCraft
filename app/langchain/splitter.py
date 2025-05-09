from langchain.text_splitter import RecursiveCharacterTextSplitter
from typing import List

from llama_index.core import Document


def split_text(text: str) -> List:
    """
    Splits the input text into chunks of specified size and overlap.

    Args:
        text (str): The text to be split.

    Returns:
        list: A list of text chunks.
    """
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=100,
        chunk_overlap=0,
        length_function=len,
        separators=["\n\n", "\n", " ", ""]
    )
    chunks = text_splitter.split_text(text)
    return chunks


if __name__ == "__main__":
    text = Document.example().text
    for chunk in split_text(text):
        print(chunk)
        print("-----")
        print("-----")