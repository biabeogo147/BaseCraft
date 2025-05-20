from typing import List
from pydantic import BaseModel

class FileCombined(BaseModel):
    path: str
    description: str
    depend_on: List[str]


class FileCombines(BaseModel):
    files: List[FileCombined]


class FileOrder(BaseModel):
    path: str
    order: int
    description: str
    depend_on: List[str]


class FileOrders(BaseModel):
    files: List[FileOrder]