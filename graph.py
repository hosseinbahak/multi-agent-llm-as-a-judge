# graph.py
import uuid
from collections import defaultdict

class ArgumentGraph:
    def __init__(self):
        self.nodes = {}  # node_id -> {role, text, confidence}
        self.edges = defaultdict(list)

    def add_node(self, role: str, text: str, confidence: float):
        node_id = str(uuid.uuid4())[:8]
        self.nodes[node_id] = {"role": role, "text": text, "confidence": confidence}
        return node_id

    def add_edge(self, parent: str, child: str):
        self.edges[parent].append(child)

    def get_leaves(self):
        leaves = []
        for nid, info in self.nodes.items():
            if nid not in self.edges or len(self.edges[nid]) == 0:
                if info["role"] != "input":
                    leaves.append(nid)
        return leaves

    def to_dict(self):
        return {"nodes": self.nodes, "edges": dict(self.edges)}
