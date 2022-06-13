import functools
from rejoice.lib import Language
from rejoice import vars


class MathLang(Language):
    """A simple Math language for testing."""

    @functools.cache
    def all_operators(self) -> "list[tuple]":
        return list(map(self.op_tuple, [
            ("Diff", "x", "y"),
            ("Integral", "x", "y"),
            ("Add", "x", "y"),
            ("Sub", "x", "y"),
            ("Mul", "x", "y"),
            ("Div", "x", "y"),
            ("Pow", "x", "y"),
            ("Ln", "x"),
            ("Sqrt", "x"),

            ("Sin", "x"),
            ("Cos", "x")
        ]))

    @functools.cache
    def all_rules(self) -> "list[list]":
        a, b, c, x, f, g, y = vars("a b c x f g y") 
        op = self.all_operators_obj()
        return [
            ["comm-add", op.add(a, b), op.add(b, a)],
            ["comm-mul", op.mul(a, b), op.mul(b, a)],
            ["assoc-add", op.add(op.add(a, b), c), op.add(a, op.add(b, c))],
            ["assoc-mul", op.mul(op.mul(a, b), c), op.mul(a, op.mul(b, c))],

            ["sub-canon",  op.sub(a, b),  op.add(a, op.mul(-1, b))],
            ["zero-add",  op.add(a, 0),  a],
            ["zero-mul",  op.mul(a, 0),  0],
            ["one-mul",   op.mul(a, 1),  a],

            ["add-zero",  a,  op.add(a, 0)],
            ["mul-one",   a,  op.mul(a, 1)],

            ["cancel-sub",  op.sub(a, a),  0],

            ["distribute",  op.mul(a, op.add(b, c)), op.add(op.mul(a, b), op.mul(a, c))],
            ["factor",      op.add(op.mul(a, b), op.mul(a, c)),  op.mul(a, op.add(b, c))],
            ["pow-mul",  op.mul(op.pow(a, b), op.pow(a, c)),  op.pow(a, op.add(b, c))],
            ["pow1",     op.pow(x, 1),  x],
            ["pow2",     op.pow(x, 2),  op.mul(x, x)],

            ["d-add",  op.diff(x, op.add(a, b)),  op.add(op.diff(x, a), op.diff(x, b))],
            ["d-mul",  op.diff(x, op.mul(a, b)),  op.add(op.mul(a, op.diff(x, b)), op.mul(b, op.diff(x, a)))],

            ["d-sin",  op.diff(x, op.sin(x)),  op.cos(x)],
            ["d-cos",  op.diff(x, op.cos(x)),  op.mul(-1, op.sin(x))],

            ["i-one",    op.integral(1, x),          x],
            ["i-cos",    op.integral(op.cos(x), x),     op.sin(x)],
            ["i-sin",    op.integral(op.sin(x), x),     op.mul(-1, op.cos(x))],
            ["i-sum",    op.integral(op.add(f, g), x),  op.add(op.integral(f, x), op.integral(g, x))],
            ["i-dif",    op.integral(op.sub(f, g), x),  op.sub(op.integral(f, x), op.integral(g, x))],
            ["i-parts",  op.integral(op.mul(a, b), x),  op.sub(op.mul(a, op.integral(b, x)), op.integral(op.mul(op.diff(x, a), op.integral(b, x)), x))],            
        ]


    def get_terminals(self) -> "list":
        return [0, 1, 2]

    def eclass_analysis(self, car, cdr) -> any:
        ops = self.all_operators_obj()
        # This could be a literal encoded in a string
        try:
            return float(car)
        except:
            pass

        # Else it is an operation with arguments
        op = car
        args = cdr

        try:
            a = float(args[0])
            b = float(args[1])
            if op == ops.add:
                return a + b
            if op == ops.sub:
                return a - b
            if op == ops.mul:
                return a * b
            if op == ops.div and b != 0.0:
                return a / b
        except:
            pass
        return None
