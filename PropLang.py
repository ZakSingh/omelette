import functools
from rejoice.lib import Language
from rejoice import vars


class PropLang(Language):
    """A simple Propositional Logic language."""

    @functools.cache
    def all_operators(self) -> "list[tuple]":
        return list(map(self.op_tuple, [
            ("And", "x", "y"),
            ("Not", "x"),
            ("Or", "x", "y"),
            ("Implies", "x", "y")
        ]))

    @functools.cache
    def all_rules(self) -> "list[list]":
        a, b, c = vars("a b c")
        op = self.all_operators_dict()
        AND, NOT, OR, IM = op["and"], op["not"], op["or"], op["implies"]
        # return [
        #     ["def_imply",        IM(a, b),                      OR(NOT(a), b)],
        #     # ["double_neg",       NOT(NOT(a)),                   a],
        #     # ["def_imply_flip",   OR(NOT(a), b),                 IM(a, b)],
        #     ["double_neg_flip",  a,                             NOT(NOT(a))],
        #     ["assoc_or",         OR(a, OR(b, c)),        OR(OR(a, b), c)],
        #     ["dist_and_or",      AND(a, OR(b, c)),  OR(AND(a, b), AND(a, c))],
        #     ["dist_or_and",      OR(a, AND(b, c)),  AND(OR(a, b), OR(a, c))],
        #     ["comm_or",          OR(a, b),                      OR(b, a)],
        #     ["comm_and",         AND(a, b),                     AND(b, a)],
        #     ["lem",              OR(a, NOT(a)),                 True],
        #     # ["or_true",          OR(a, True),                   True],
        #     ["and_true",         AND(a, True),                  a],
        #     # ["contrapositive",   IM(a, b),          IM(NOT(b), NOT(a))],
        #     # ["lem_imply",        AND(IM(a, b), IM(NOT(a), c)),  OR(b, c)],
        # ]
        return [
            ["def_imply",        IM(a, b),                      OR(NOT(a), b)],
            ["double_neg",       NOT(NOT(a)),                   a],
            ["def_imply_flip",   OR(NOT(a), b),                 IM(a, b)],
            ["double_neg_flip",  a,                             NOT(NOT(a))],
            ["assoc_or",         OR(a, OR(b, c)),        OR(OR(a, b), c)],
            ["dist_and_or",      AND(a, OR(b, c)),  OR(AND(a, b), AND(a, c))],
            ["dist_or_and",      OR(a, AND(b, c)),  AND(OR(a, b), OR(a, c))],
            ["comm_or",          OR(a, b),                      OR(b, a)],
            ["comm_and",         AND(a, b),                     AND(b, a)],
            ["lem",              OR(a, NOT(a)),                 True],
            ["or_true",          OR(a, True),                   True],
            ["and_true",         AND(a, True),                  a],
            ["contrapositive",   IM(a, b),          IM(NOT(b), NOT(a))],
            # ["lem_imply",        AND(IM(a, b), IM(NOT(a), c)),  OR(b, c)],
        ]

    def get_terminals(self) -> "list":
        return [True, False]

    def eclass_analysis(self, car, cdr) -> any:
        op = self.all_operators_dict()
        AND, NOT, OR, IM = op["and"], op["not"], op["or"], op["implies"]
        # This could be a literal
        if type(car) == bool:
            return car

        # Or a variable
        if len(cdr) == 0:
            return None

        # Else it is an operation with arguments
        op = car
        args = cdr

        # Symbolic values cannot be evaluated
        if any(type(a) != bool for a in args):
            return None

        a = args[0]
        if op == NOT:
            return not a

        b = args[1]
        if op == AND:
            return a and b

        if op == OR:
            return a or b

        if op == IM:
            return a or not b

        return None
