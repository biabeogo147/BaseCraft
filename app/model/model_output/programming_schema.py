from typing import List
from pydantic import BaseModel

class File(BaseModel):
    path: str
    content: str