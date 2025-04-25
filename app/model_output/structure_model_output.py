from typing import List
from pydantic import BaseModel

class FileDescription(BaseModel):
    path: str
    description: str

class DirectoryDescription(BaseModel):
    directories: List[str]
    files: List[FileDescription]