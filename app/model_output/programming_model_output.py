from typing import List
from pydantic import BaseModel

class File(BaseModel):
    path: str
    content: str 

class DirectoryStructure(BaseModel):
    directories: List[str]
    files: List[File]