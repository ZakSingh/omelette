import functools

# from .rejoice import *
from rejoice import *
from typing import Protocol, Union, NamedTuple
from collections import OrderedDict
import torch
import torch_geometric as geom
import networkx as nx
import matplotlib.pyplot as plt
import numpy as np
import torch_geometric.transforms as T
import time
import sys

# needed for safe expression generation
sys.setrecursionlimit(10**5)


class ObjectView(object):
    def __init__(self, d):
        self.__dict__ = d


class Language(Protocol):
    """A base Language for an equality saturation task. This will be passed to egg."""
    @property
    def name(self):
        return type(self).__name__

    def op_tuple(self, op):
        """Convert an operator string (name, x, y, ...) into a named tuple"""
        name, *args = op
        tup = NamedTuple(name, [(a, int) for a in args])
        globals()[name] = tup
        return tup

    def eclass_analysis(self, *args) -> any:
        ...

    def all_operators(self) -> list[tuple]:
        ...

    @property
    def num_actions(self) -> int:
        return len(self.all_rules())

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

    @property
    def num_node_features(self) -> int:
        return 4 + len(self.all_operators())

    def get_feature_upper_bounds(self):
        return np.array([1, 1, 1, np.inf] + ([1] * len(self.all_operators())))

    def encode_node(self, operator: Union[int, tuple]) -> torch.Tensor:
        """[is_eclass, is_enode, is_scalar, scalar_val, ...onehot_optype]"""
        onehot = torch.zeros(self.num_node_features)

        if isinstance(operator, int):
            onehot[2] = 1
            onehot[3] = operator
            return onehot

        # is an enode
        onehot[1] = 1

        for ind, op in enumerate(self.all_operators()):
            if isinstance(operator, op):
                onehot[4 + ind] = 1
                return onehot

        raise Exception("Failed to encode node")

    def feature_names(self):
        features = ["is_eclass",
                    "is_enode",
                    "is_scalar",
                    "scalar_val"]

        op_names = [op.__name__ for op in self.all_operators()]
        return features + op_names

    def decode_node(self, node: torch.Tensor):
        dnode = {"type": "eclass" if node[0] == 1 else "enode", "is_scalar": bool(node[2]), "value": node[3]}
        # if it's an enode op, find its op type
        if node[1] == 1 and node[2] == 0:
            all_ops = self.all_operators()
            ind = torch.argmax(torch.Tensor(node[4:])).item()
            op = all_ops[ind]
            dnode["op"] = op.__name__
        return dnode

    def encode_eclass(self, eclass_id: int, data: any) -> torch.Tensor:
        onehot = torch.zeros(self.num_node_features)
        onehot[0] = 1
        return onehot

    @functools.cached_property
    def op_to_ind(self):
        op_to_ind_table = {}
        for ind, op in enumerate(self.all_operators()):
            op_to_ind_table[op] = ind
        return op_to_ind_table

    def gen_expr(self, root_op=None, p_leaf=0.8):
        """Generate an arbitrary expression which abides by the language."""
        ops = self.all_operators()
        root = np.random.choice(ops) if root_op is None else root_op
        children = []
        for i in range(len(root._fields)):
            if np.random.uniform(0, 1) < p_leaf:
                children.append(np.random.randint(0, 2))
            else:
                chosen_op = np.random.choice(ops)
                op_children = []
                for j in range(len(chosen_op._fields)):
                    op_children.append(self.gen_expr(chosen_op))
                children.append(chosen_op(*op_children))
        return root(*children)

    def encode_egraph(self, egraph: EGraph, y=None) -> geom.data.Data:
        # first_stamp = int(round(time.time() * 1000))
        num_enodes = egraph.num_enodes()
        eclass_ids = egraph.eclass_ids()
        num_eclasses = len(eclass_ids)
        enode_eclass_edges = torch.zeros([2, num_enodes])
        x = torch.zeros([num_eclasses + num_enodes, self.num_node_features])
        x[:num_eclasses, 0] = 1  # make eclass nodes
        x[num_eclasses:, 1] = 1  # mark enodes

        curr = num_eclasses
        edge_curr = 0

        eclass_to_ind = dict(zip(eclass_ids, range(num_eclasses)))
        classes = egraph.classes()

        all_node_edges = []
        # print("enodes", num_enodes, "eclasses", num_eclasses)

        for eclass_id, (data, nodes) in classes.items():
            eclass_ind = eclass_to_ind[eclass_id]
            num_eclass_nodes = len(nodes)
            # create edges from eclass to member enodes
            enode_eclass_edges[0, edge_curr:(edge_curr + num_eclass_nodes)] = eclass_ind
            enode_eclass_edges[1, edge_curr:(edge_curr + num_eclass_nodes)] = torch.arange(curr, curr + num_eclass_nodes)
            edge_curr = edge_curr + num_eclass_nodes

            for node in nodes:
                if isinstance(node, int):
                    x[curr, 2] = 1
                    x[curr, 3] = node
                else:
                    x[curr, 4 + self.op_to_ind[type(node)]] = 1  # encode operator type
                    # connect to child eclasses
                    if isinstance(node, tuple):
                        all_node_edges.append(torch.stack([torch.full([len(node)], curr),
                                              torch.Tensor([eclass_to_ind[str(ecid)] for ecid in node])]))
                curr += 1

        edge_index = torch.concat([enode_eclass_edges, *all_node_edges], dim=1).long()
        edge_index, _ = geom.utils.add_remaining_self_loops(edge_index)
        data = geom.data.Data(x=x, edge_index=edge_index, y=y)
        # second_stamp = int(round(time.time() * 1000))
        # Calculate the time taken in milliseconds
        # time_taken = second_stamp - first_stamp
        # print("time_taken", time_taken, data)
        return data
    #
    # def encode_egraph(self, egraph: EGraph) -> geom.data.Data:
    #     """Encode an egg egraph into a Pytorch Geometric data object"""
    #     # first_stamp = int(round(time.time() * 1000))
    #
    #     classes = egraph.classes()
    #     all_nodes = []
    #     all_edges = []
    #     edge_attr = []
    #     eclass_to_ind = {}
    #
    #     # Insert eclasses first as enodes will refer to them
    #     for eclass_id, (data, nodes) in classes.items():
    #         all_nodes.append(self.encode_eclass(eclass_id, data))
    #         eclass_to_ind[eclass_id] = len(all_nodes) - 1
    #
    #     for eclass_id, (data, nodes) in classes.items():
    #         for node in nodes:
    #             all_nodes.append(self.encode_node(node))
    #
    #             # connect each node to its eclass
    #             all_edges.append(torch.Tensor([eclass_to_ind[eclass_id], len(all_nodes) - 1]))
    #             edge_attr.append(torch.Tensor([0, 1]))
    #
    #             # connect each node to its child eclasses
    #             if isinstance(node, tuple):
    #                 for ecid in node:
    #                     all_edges.append(torch.Tensor([len(all_nodes) - 1, eclass_to_ind[str(ecid)]]))
    #                     edge_attr.append(torch.Tensor([1, 0]))
    #
    #     x = torch.stack(all_nodes, dim=0)
    #     edge_index = torch.stack(all_edges, dim=0).T.long()
    #     edge_attr = torch.stack(edge_attr, dim=0)
    #     # edge_index, edge_attr = geom.utils.add_remaining_self_loops(edge_index, edge_attr)
    #     data = geom.data.Data(x=x, edge_index=edge_index, edge_attr=edge_attr)
    #     # second_stamp = int(round(time.time() * 1000))
    #
    #     # time_taken = second_stamp - first_stamp
    #     # print("time_taken", time_taken)
    #     return data

    def viz_egraph(self, data):
        """Vizualize a PyTorch Geometric data object containing an egraph."""
        g = geom.utils.to_networkx(data, node_attrs=['x'])

        for u, data in g.nodes(data=True):
            decoded = self.decode_node(data["x"])
            if decoded["type"] == "eclass":
                data['name'] = "eclass"
            elif decoded["is_scalar"]:
                data['name'] = decoded["value"]
            else:
                data["name"] = decoded["op"]
            del data['x']

        node_labels = {}
        for u, data in g.nodes(data=True):
            node_labels[u] = data['name']

        pos = nx.nx_agraph.graphviz_layout(g, prog="dot")
        nx.draw(g, labels=node_labels, pos=pos)
        plt.show()
        return g
