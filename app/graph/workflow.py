from langgraph.graph import StateGraph, END
from app.core.state import AgentState
from app.graph.nodes import AgentNodes

def create_workflow(nodes: AgentNodes):
    workflow = StateGraph(AgentState)

    # Add nodes
    workflow.add_node("detect_language", nodes.detect_language)
    workflow.add_node("retrieve_documents", nodes.retrieve_documents)
    workflow.add_node("grade_documents", nodes.grade_documents)
    workflow.add_node("rewrite_query", nodes.rewrite_query)
    workflow.add_node("generate_answer", nodes.generate_answer)

    # Define conditional logic
    def decide_to_generate(state: AgentState):
        grade = state.get("grade")
        retry_count = state.get("retry_count", 0)
        
        if grade == "relevant" or retry_count >= 3:
            return "generate_answer"
        else:
            return "rewrite_query"

    # Define edges
    workflow.set_entry_point("detect_language")
    workflow.add_edge("detect_language", "retrieve_documents")
    workflow.add_edge("retrieve_documents", "grade_documents")
    
    workflow.add_conditional_edges(
        "grade_documents",
        decide_to_generate,
        {
            "generate_answer": "generate_answer",
            "rewrite_query": "rewrite_query"
        }
    )
    
    workflow.add_edge("rewrite_query", "retrieve_documents")
    workflow.add_edge("generate_answer", END)

    return workflow.compile()
