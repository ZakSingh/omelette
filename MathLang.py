import functools
from rejoice.lib import Language
from rejoice import vars


class MathLang(Language):
    """A simple Math language for testing."""

    @functools.cache
    def all_operators(self) -> list[tuple]:
        return list(map(self.op_tuple, [
            ("Add", "x", "y"),
            ("Mul", "x", "y")
        ]))

    @functools.cache
    def all_rules(self) -> list[list]:
        a, b = vars("a b")
        op = self.all_operators_obj()
        return [
                ["commute-add", op.add(a, b), op.add(b, a)],
                ["commute-mul", op.mul(a, b), op.mul(b, a)],
                ["add-0", op.add(a, 0), a],
                ["mul-0", op.mul(a, 0), 0],
                ["mul-1", op.mul(a, 1), a]
             ]

    def eclass_analysis(self, *args) -> any:
        op = self.all_operators_obj()
        # in MathLang, every eclass has at most 2 operands
        car, cdr = args

        # This could be a literal encoded in a string
        try:
            return float(car)
        except:
            pass

        # Else it is an operation with arguments
        operator = car
        args = cdr
        try:
            a = float(args[0])
            b = float(args[1])
            if operator == op.add:
                return a + b
            if operator == op.mul:
                return a * b
        except:
            pass

        return None
