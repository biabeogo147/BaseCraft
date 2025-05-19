from typing import List
from pydantic import BaseModel

class FileRequirement(BaseModel):
    path: str
    depend_on: List[str] = []

class HierarchyDirectory(BaseModel):
    directories: List[str]
    files: List[FileRequirement]