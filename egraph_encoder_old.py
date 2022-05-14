import torch
from torch_geometric.data import Data
import json
from itertools import combinations, groupby
from enum import Enum

class EGraphEncoder:
    """This class handles conversion of egg's egraph data structure into a PyTorch Geometric Data object."""

    class EGraphEdgeType(Enum):
        eclass = 0
        child = 1

    def __init__(self, egraph):
        self.egraph = egraph
        self.enodes = self.collect_enodes()
        # in the future, all_ops should come from the LANGUAGE DEFINITION.
        # right now this is NOT GENERALIZABLE if every graph doesn't use
        # the same exact set of operator types.
        self.all_ops = list(set([node["op"] for node in self.enodes]))
        self.all_ops.sort()

    def build_enode(self, enode):
        """Converts enode from native list format to dict"""
        enode[0]["eclass"] = enode[1]
        return enode[0]

    def collect_enodes(self):
        """Get a list of all enodes in the egraph"""
        nodes = []
        for eclass in self.egraph["classes"].values():
            nodes += [self.build_enode([node, eclass["id"]]) for node in eclass["nodes"]]
            nodes += [self.build_enode(n) for n in eclass["parents"]]
        # remove duplicates (hacky...)
        nodes = list(map(json.loads, list(set([json.dumps(n) for n in nodes]))))
        # create a canonical ordering of nodes
        nodes = sorted(nodes, key=lambda n: (n["eclass"], n["op"], n["children"]))
        return nodes

    def enode_to_ind(self, enode):
        """Lookup the index of a specific enode"""
        for ind, enode_i in enumerate(self.enodes):
            if enode == enode_i:
                return ind
        raise IndexError

    def encode_egraph(self):
        """Encodes an EGraph into a PyTorch Geometric Data object"""
        # build node features
        x = torch.zeros((len(self.enodes), len(self.all_ops)))
        for ind, node in enumerate(self.enodes):
            x[ind][self.all_ops.index(node["op"])] = 1

        edges = []
        edge_attrs = []

        # for each class, connect every node to eachother with an edge of type "eclass"
        for k, eclass in groupby(self.enodes, lambda e: e["eclass"]):
            pairs = [torch.Tensor(list(map(self.enode_to_ind, enode_pair)))
                     for enode_pair in list(combinations(eclass, 2))]
            edges += pairs
            edge_attrs += torch.Tensor([self.EGraphEdgeType.eclass.value] * len(pairs))

        # now create edges between enode -> all members of child eclass
        for ind, node in enumerate(self.enodes):
            children = filter(lambda n: n["eclass"] in node["children"], self.enodes)
            pairs = [torch.Tensor([ind, self.enode_to_ind(c)]) for c in children]
            edges += pairs
            edge_attrs += torch.Tensor([self.EGraphEdgeType.child.value] * len(pairs))

        edge_index = torch.stack(edges, dim=0).T.long()
        edge_attrs = torch.stack(edge_attrs)

        return Data(x=x, edge_index=edge_index, edge_attr=edge_attrs)
