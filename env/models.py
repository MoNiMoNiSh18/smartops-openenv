from pydantic import BaseModel
from typing import Optional, List, Literal

class Observation(BaseModel):
    ticket_id: str
    user_message: str
    category: Optional[str]
    priority: Optional[str]
    history: List[str]
    time_elapsed: int

class Action(BaseModel):
    action_type: Literal["classify", "prioritize", "respond", "resolve"]
    content: Optional[str]