from langgraph.graph import StateGraph, END
from langgraph.checkpoint.memory import MemorySaver
from app.core.state import AgentState
from app.graph.nodes import AgentNodes

def create_workflow(nodes: AgentNodes):
    memory = MemorySaver()
    workflow = StateGraph(AgentState)

    # Add nodes
    workflow.add_node("detect_language", nodes.detect_language)
    workflow.add_node("contextualize_query", nodes.contextualize_query)
    workflow.add_node("retrieve_documents", nodes.retrieve_documents)
    workflow.add_node("grade_documents", nodes.grade_documents)
    workflow.add_node("rewrite_query", nodes.rewrite_query)
    workflow.add_node("generate_answer", nodes.generate_answer)
    workflow.add_node("clarify_ambiguity", nodes.clarify_ambiguity)

    # Define conditional logic
    def decide_to_generate(state: AgentState):
        grade = state.get("grade")
        retry_count = state.get("retry_count", 0)

        if grade == "relevant":
            return "generate_answer"
        elif retry_count >= 1: # Stricter: Only allow 1 rewrite attempt
            return "clarify_ambiguity"
        else:
            return "rewrite_query"

    # Define edges
    workflow.set_entry_point("detect_language")
    workflow.add_edge("detect_language", "contextualize_query")
    workflow.add_edge("contextualize_query", "retrieve_documents")
    workflow.add_edge("retrieve_documents", "grade_documents")

    workflow.add_conditional_edges(
        "grade_documents",
        decide_to_generate,
        {
            "generate_answer": "generate_answer",
            "rewrite_query": "rewrite_query",
            "clarify_ambiguity": "clarify_ambiguity"
        }
    )

    workflow.add_edge("rewrite_query", "retrieve_documents")
    workflow.add_edge("generate_answer", END)
    workflow.add_edge("clarify_ambiguity", END)

    return workflow.compile(checkpointer=memory)
