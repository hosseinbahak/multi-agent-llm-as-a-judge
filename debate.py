# debate.py
from llm_simulator import simulate_llm

def mcts_expand(graph, parent_id: str, role: str):
    """
    Expand the reasoning graph from a given parent node.
    """
    parent_text = graph.nodes[parent_id]["text"]
    resp, conf = simulate_llm(role, parent_text)
    cid = graph.add_node(role, resp, conf)
    graph.add_edge(parent_id, cid)
    return cid

def debate_round(graph, round_num: int = 1):
    """
    One debate round: expand all leaves with all judge roles.
    """
    leaves = graph.get_leaves()
    for leaf in leaves:
        for role in ["logician", "innovator", "synthesizer"]:
            mcts_expand(graph, leaf, role)
