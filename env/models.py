from pydantic import BaseModel
from typing import  List,Literal,Optional

class Observation(BaseModel):
    ticket_id: str
    user_message: str
    category: str | None
    priority: str | None
    history: list
    time_elapsed: int

class Action(BaseModel):
    action_type: Literal["classify", "prioritize", "respond", "resolve"]
    content: Optional[str]