from typing import Annotated, List, Dict, Any, Optional
from typing_extensions import TypedDict
from langchain_core.messages import BaseMessage
import operator

class AgentState(TypedDict):
    """
    State of the agent graph.
    """
    messages: Annotated[List[BaseMessage], operator.add]
    query: str
    language: str
    documents: List[Dict[str, Any]]
    answer: Optional[str]
    error: Optional[str]
    grade: Optional[str]
    retry_count: int
