from .rejoice import *
from typing import Protocol, Union, NamedTuple
import torch


class ObjectView(object):
    def __init__(self, d):
        self.__dict__ = d


class Language(Protocol):

    def op_tuple(self, op):
        name, *args = op
        return NamedTuple(name, [(a, int) for a in args])

    def eclass_analysis(self, *args) -> any:
        ...

    def all_operators(self) -> list[tuple]:
        ...

    def all_operators_obj(self):
        op_dict = dict([(operator.__name__.lower(), operator) for operator in self.all_operators()])
        return ObjectView(op_dict)

    def all_rules(self) -> list[list]:
        ...

    def rewrite_rules(self):
        rules = list()
        for rl in self.all_rules():
            name = rl[0]
            frm = rl[1]
            to = rl[2]
            rules.append(Rewrite(frm, to, name))
        return rules

    def num_node_features(self) -> int:
        return 3 + len(self.all_operators())

    def encode_node(self, operator: Union[int, tuple]) -> torch.Tensor:
        onehot = torch.zeros(self.num_node_features())

        if isinstance(operator, int):
            onehot[1] = 1
            onehot[2] = operator
            return onehot

        for ind, op in enumerate(self.all_operators()):
            if isinstance(operator, op):
                onehot[3 + ind] = 1
                return onehot

        raise Exception("Failed to encode node")

    def decode_node(self, node: torch.Tensor):
        dnode = {"type": "eclass" if node[0] == 1 else "enode", "is_scalar": bool(node[1]), "value": node[2]}
        if node[0] == 0 and node[1] == 0:
            all_ops = self.all_operators()
            ind = torch.argmax(torch.Tensor(node[3:])).item()
            op = all_ops[ind]
            dnode["op"] = op.__name__
        return dnode

    def encode_eclass(self, eclass_id: int, data: any) -> torch.Tensor:
        onehot = torch.zeros(self.num_node_features())
        onehot[0] = 1
        return onehot

    def encode_egraph(self, egraph: EGraph) -> tuple[torch.Tensor, torch.Tensor]:
        classes = egraph.classes()
        print(classes)
        all_nodes = []
        all_edges = []
        eclass_to_ind = {}

        # Insert eclasses first as enodes will refer to them
        for eclass_id, (data, nodes) in classes.items():
            all_nodes.append(self.encode_eclass(eclass_id, data))
            eclass_to_ind[eclass_id] = len(all_nodes) - 1

        for eclass_id, (data, nodes) in classes.items():
            for node in nodes:
                print("node", node)
                all_nodes.append(self.encode_node(node))

                # connect each node to its eclass
                all_edges.append(torch.Tensor([eclass_to_ind[eclass_id], len(all_nodes) - 1]))

                # connect each node to its child eclasses
                if isinstance(node, tuple):
                    for ecid in node:
                        all_edges.append(torch.Tensor([len(all_nodes) - 1, eclass_to_ind[str(ecid)]]))

        x = torch.stack(all_nodes, dim=0)
        edge_index = torch.stack(all_edges, dim=0).T.long()

        return x, edge_index
