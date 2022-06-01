import functools

# from .rejoice import *
from rejoice import *
from typing import Protocol, Union, NamedTuple
from collections import OrderedDict
from rejoice.util import BytesIntEncoder
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

    num_static_features = 4

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

    def all_operators(self) -> "list[tuple]":
        ...

    @property
    def num_actions(self) -> int:
        return len(self.all_rules())

    def all_operators_obj(self):
        op_dict = self.all_operators_dict()
        return ObjectView(op_dict)

    def all_operators_dict(self):
        op_dict = dict([(operator.__name__.lower(), operator)
                       for operator in self.all_operators()])
        return op_dict

    def all_rules(self) -> "list[list]":
        ...

    def get_terminals(self) -> "list":
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
        return self.num_static_features + len(self.get_terminals()) + len(self.all_operators())

    def get_feature_upper_bounds(self):

        return np.array(([1] * self.num_static_features) + ([1] * len(self.get_terminals())) + ([1] * len(self.all_operators())))

    def feature_names(self):
        features = ["is_eclass",
                    "is_enode",
                    "is_scalar",
                    "is_terminal"]
        terminal_names = [str(t) for t in self.get_terminals()]
        op_names = [op.__name__ for op in self.all_operators()]
        return features + terminal_names + op_names

    def decode_node(self, node: torch.Tensor):
        dnode = {"type": "eclass" if node[0] == 1 else "enode",
                 "is_scalar": bool(node[2]),
                 "is_terminal": bool(node[3]), }
        # if it's an enode op, find its op type
        if node[1] == 1 and node[2] == 0 and node[3] == 0:
            all_ops = self.all_operators()
            op_ind = torch.argmax(torch.Tensor(
                node[self.num_static_features + len(self.get_terminals()):])).item()
            op = all_ops[op_ind]
            dnode["op"] = op.__name__
        return dnode

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
            enode_eclass_edges[0, edge_curr:(
                edge_curr + num_eclass_nodes)] = eclass_ind
            enode_eclass_edges[1, edge_curr:(
                edge_curr + num_eclass_nodes)] = torch.arange(curr, curr + num_eclass_nodes)
            edge_curr = edge_curr + num_eclass_nodes

            for node in nodes:
                # we only want to encode if they're terminals... everything else will cause learning confusion.
                if isinstance(node, int) or isinstance(node, float) or isinstance(node, bool) or isinstance(node, str):
                    try:
                        term_ind = self.get_terminals().index(node)
                        x[curr, 3] = 1
                        x[curr, self.num_static_features + term_ind] = 1
                    except ValueError:
                        # it's an unknown scalar (not in terminals list)
                        x[curr, 2] = 1
                else:
                    # encode operator type
                    x[curr, self.num_static_features +
                        len(self.get_terminals()) + self.op_to_ind[type(node)]] = 1
                    # connect to child eclasses
                    if isinstance(node, tuple):
                        all_node_edges.append(torch.stack([torch.full([len(node)], curr),
                                              torch.Tensor([eclass_to_ind[str(ecid)] for ecid in node])]))
                curr += 1

        edge_index = torch.concat(
            [enode_eclass_edges, *all_node_edges], dim=1).long()
        edge_index, _ = geom.utils.add_remaining_self_loops(edge_index)
        data = geom.data.Data(x=x, edge_index=edge_index, y=y)
        # second_stamp = int(round(time.time() * 1000))
        # Calculate the time taken in milliseconds
        # time_taken = second_stamp - first_stamp
        # print("time_taken", time_taken, data)
        return data

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
