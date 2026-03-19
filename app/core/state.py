from typing import List, Dict, Any, Optional
from typing_extensions import TypedDict
from langchain_core.messages import BaseMessage

class AgentState(TypedDict):
    """
    State of the agent graph.
    """
    messages: List[BaseMessage]
    query: str
    language: str
    documents: List[Dict[str, Any]]
    answer: Optional[str]
    error: Optional[str]
    grade: Optional[str]
    retry_count: int
    contextualized_query: Optional[str]
