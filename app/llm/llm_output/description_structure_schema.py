from typing import List
from pydantic import BaseModel

class FileDescription(BaseModel):
    path: str
    description: str

class FileDescriptions(BaseModel):
    files: List[FileDescription]