import functools
from rejoice.lib import Language, TestExprs
from rejoice import vars


class LambdaLang(Language):
    """A simple Lambda language for testing."""

    def get_supported_datatypes(self):
        return ["symbols"]

    @functools.cache
    def all_operators(self) -> "list[tuple]":
        return list(map(self.op_tuple, [
            ("f", "x"),
            ("g", "x", "y"),
            ("h", "x")
        ]))

    @functools.cache
    def all_rules(self) -> "list[list]":
        x, y = vars("x y") 
        Z = "Z"
        op = self.all_operators_obj()
        f, g, h = op.f, op.g, op.h

        return [
            ["h",  h(g(x, Z)),  h(g(f(x), Z))],
            ["g1", g(f(x), y),  g(x, f(y))],
            ["g2", g(Z, y),     g(Z, Z)],
            ["g3", g(x, y), x]
        ]


    def get_terminals(self) -> "list":
        return ["Z"]

    def eclass_analysis(self, car, cdr) -> any:
        return None

    def get_single_task_exprs(self):
        ops = self.all_operators_obj()
        f, g, h = ops.f, ops.g, ops.h

        s = h(g("x", "Z"))
        e = h(g("x", "Z"))

        return TestExprs(saturatable=s,
                         explodes=e)

    def get_multi_task_exprs(self, count=16):
        """Get a list of exprs for use in multi-task RL training"""
        return [self.gen_expr(p_leaf=0.0) for i in range(count)]
