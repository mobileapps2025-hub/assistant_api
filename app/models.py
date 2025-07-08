from pydantic import BaseModel, Field
from typing import Any, Dict, List, Optional # Added List, Optional

# --- Pydantic Models ---

class ContentItem(BaseModel):
    text: str
    type: str

class Message(BaseModel):
    role: str
    content: Optional[List[ContentItem] | str] = None
    tool_call_id: Optional[str] = None 
    name: Optional[str] = None
    tool_calls: Optional[List[Any]] = None 
    anotations: Optional[str] = None 

class ChatRequest(BaseModel):
    messages: list[Message]

# -- events request model --
class EventsBetweenWeeksRequest(BaseModel):
    storeId: str
    startingWeek: int
    endingWeek: int
    year: int
