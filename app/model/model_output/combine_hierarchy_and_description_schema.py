from typing import List
from pydantic import BaseModel

class FileCombined(BaseModel):
    path: str
    description: str
    depend_on: List[str]

class DirectoryCombined(BaseModel):
    directories: List[str]
    files: List[FileCombined]


class FileOrder(BaseModel):
    path: str
    order: int
    description: str
    depend_on: List[str]


class DirectoryOrder(BaseModel):
    directories: List[str]
    files: List[FileOrder]