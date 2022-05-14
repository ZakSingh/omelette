from typing import NamedTuple, Union
from rejoice.util import ObjectView
import torch


def op_tuple(pattern):
    name, *args = pattern
    return NamedTuple(name, [(a, int) for a in args])


# canonical ordering of operators (needed for stable one-hot encoding)
oplist = list(map(op_tuple, [
    ("Add", "x", "y"),
    ("Mul", "x", "y")
]))


def is_leaf(op) -> bool:
    if isinstance(op, str) or isinstance(op, int) or isinstance(op, float):
        return True
    return False


def onehot_op(operator: Union[tuple, int]) -> torch.Tensor:
    """Converts a given operator to its one-hot encoding."""
    num_operators = len(oplist) + 2
    onehot = torch.zeros(num_operators)

    if is_leaf(operator):
        onehot[0] = 1
        return onehot

    for ind, op in enumerate(oplist):
        if isinstance(operator, op):
            onehot[ind + 2] = 1
            return onehot

    raise Exception("Couldn't identify operator")


def op_from_onehot(onehot: torch.Tensor):
    """Retrieve a pattern tuple from its one-hot encoding"""
    ind = torch.argmax(onehot).item() - 1
    if ind == 0:
        return None  # leaf node
    return oplist[ind]


# dict of op_type -> op_tuple
op_dict = dict([(operator.__name__.lower(), operator) for operator in oplist])
# Use this version for easy dot notation access
operators = ObjectView(op_dict)
