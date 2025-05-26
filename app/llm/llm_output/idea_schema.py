from typing import List
from pydantic import BaseModel


class Idea(BaseModel):
    project_name: str
    project_goal: str
    key_features: List[str]
    system_architecture: str
    suitable_technologies: List[str]
