from typing import List
from pydantic import BaseModel

class FileCombined(BaseModel):
    path: str
    description: str
    depend_on: List[str]

class DirectoryCombined(BaseModel):
    directories: List[str]
    files: List[FileCombined]