from langchain_core.messages import HumanMessage
from langgraph.checkpoint.memory import MemorySaver
from langgraph.graph import END
from langgraph.graph import START
from langgraph.graph import StateGraph

from .nodes import compress_context_node
from .nodes import planning_agent_node
from .nodes import should_compress_check
from .state import PlanningState


def create_graph():
    graph = StateGraph(PlanningState)

    graph.add_node("compress", compress_context_node)
    graph.add_node("agent", planning_agent_node)

    graph.add_conditional_edges(
        START, should_compress_check, {"compress": "compress", "agent": "agent"}
    )
    graph.add_edge("compress", "agent")
    graph.add_edge("agent", END)

    memory = MemorySaver()
    return graph.compile(checkpointer=memory)


def get_response(graph, user_input: str, thread_id: str):
    config = {"configurable": {"thread_id": thread_id}}

    result = graph.invoke({"messages": [HumanMessage(content=user_input)]}, config)

    return result


def get_conversation_state(graph, thread_id: str) -> PlanningState | None:
    config = {"configurable": {"thread_id": thread_id}}
    try:
        state = graph.get_state(config)
        return state.values if state else None
    except Exception:
        return None
