from langgraph.graph import END, START, StateGraph

from . import nodes
from .state import InvState


def build_graph(toolbox):
    async def investigate(state):
        return await nodes.investigate(state, toolbox)

    g = StateGraph(InvState)
    g.add_node("investigate", investigate)
    g.add_node("retrieve", nodes.retrieve)
    g.add_node("draft", nodes.draft)
    g.add_node("verify", nodes.verify)

    g.add_edge(START, "investigate")
    g.add_edge("investigate", "retrieve")
    g.add_edge("retrieve", "draft")
    g.add_edge("draft", "verify")
    g.add_conditional_edges("verify", nodes.after_verify, {"draft": "draft", "end": END})
    return g.compile()
