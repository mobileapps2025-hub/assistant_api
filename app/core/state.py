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
    visual_aids: List[Dict[str, Any]]
    answer: Optional[str]
    error: Optional[str]
    grade: Optional[str]
    retry_count: int
    contextualized_query: Optional[str]
    # Grading confidence metadata (set by grade_documents)
    relevant_count: Optional[int]
    total_graded: Optional[int]
    # Per-retry search strategy override (set by rewrite_query)
    search_alpha_override: Optional[float]
