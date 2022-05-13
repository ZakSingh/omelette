from snake_egg import EGraph, Rewrite, Var, vars
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.data import Data
import numpy as np
import unittest
import json
from collections import namedtuple
from itertools import combinations, groupby
from enum import Enum
from egraph_encoder import EGraphEncoder

add = namedtuple("Add", "x y")
mul = namedtuple("Mul", "x y")

# Rewrite rules
a, b = vars("a b")
list_rules = [
    ["commute-add", add(a, b), add(b, a)],
    ["commute-mul", mul(a, b), mul(b, a)],
    ["add-0", add(a, 0), a],
    ["mul-0", mul(a, 0), 0],
    ["mul-1", mul(a, 1), a],
]

# Turn the lists into rewrites
rules = list()
for l in list_rules:
    name = l[0]
    frm = l[1]
    to = l[2]
    rules.append(Rewrite(frm, to, name))


def simplify(expr, iters=7):
    egraph = EGraph()
    egraph.add(expr)
    egraph.run(rules, iters)
    best = egraph.extract(expr)
    return best


def main():
    f = open("exgraph.json")
    egraph = json.load(f)
    encoder = EGraphEncoder(egraph)
    data = encoder.encode_egraph()
    print(data)


if __name__ == "__main__":
    main()
