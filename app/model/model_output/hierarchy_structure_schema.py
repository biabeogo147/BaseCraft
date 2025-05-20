from typing import List
from pydantic import BaseModel

class FileRequirement(BaseModel):
    path: str
    depend_on: List[str]

class FileRequirements(BaseModel):
    files: List[FileRequirement]