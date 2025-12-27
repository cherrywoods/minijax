# Copyright (c) 2025 by David Boetius
# Licensed under the MIT Licensed.
from abc import ABC, abstractmethod
from collections.abc import Iterable, Sequence
from dataclasses import dataclass


class Value(ABC):
    def __init__(self, interpreter, shape) -> None:
        self.interpreter = interpreter
        self.shape = shape

    def __neg__(self):
        return neg(self)

    def __add__(self, other):
        return add(self, other)

    def __sub__(self, other):
        return sub(self, other)

    def __mul__(self, other):
        return mul(self, other)

    def __truediv__(self, other):
        return div(self, other)

    def __matmul__(self, other):
        return dot(self, other)


class Interpreter[V: Value](ABC):
    def __init__(self, level: int):
        self.level = level

    @abstractmethod
    def wrap(self, value: Value) -> V:
        raise NotImplementedError()

    @abstractmethod
    def process(self, primitive, values: list[V], options: dict) -> V:
        raise NotImplementedError()


def new_interpreter[I: Interpreter](interpreter_cls: type[I], in_values: Iterable[Value]) -> I:
    top_level = max([val.interpreter.level for val in in_values])
    return interpreter_cls(top_level + 1)


def top_interpreter(values: Iterable[Value]):
    return max([val.interpreter for val in values], key=lambda i: i.level)


@dataclass(frozen=True)
class Primitive:
    name: str
    num_args: int
    keyword_args: tuple[str, ...] = ()

    def __call__(self, *values, **options):
        assert len(values) >= self.num_args, "Wrong number of arguments."
        options |= dict(zip(self.keyword_args, values[self.num_args :], strict=False))
        values = values[: self.num_args]
        assert set(options.keys()) == set(self.keyword_args), "Wrong keyword arguments."
        interpreter = top_interpreter(values)
        values = [interpreter.wrap(val) for val in values]
        return interpreter.process(self, values, options)


class ReduceSumPrimitive(Primitive):
    def __init__(self):
        super().__init__("reduce_sum", 1, ("axes",))

    def __call__(self, x, axes: int | Sequence[int] | None = None, keepaxes: bool = False):
        if axes is None:
            axes = tuple(range(len(x.shape)))
        elif isinstance(axes, int):
            axes = (axes,)
        res = super().__call__(x, axes=axes)
        return expand_dims(res, axes) if keepaxes else res


neg = Primitive("neg", 1)
add = Primitive("add", 2)
dot = Primitive("dot", 2)
mul = Primitive("mul", 2)
reciprocal = Primitive("reciprocal", 1)
relu = Primitive("relu", 1)
square = Primitive("square", 1)
sqrt = Primitive("sqrt", 1)
exp = Primitive("exp", 1)
log = Primitive("log", 1)
where = Primitive("where", 3)
expand_dims = Primitive("expand_dims", 1, ("axes",))
moveaxis = Primitive("moveaxis", 1, ("source", "destination"))
reshape = Primitive("reshape", 1, ("new_shape",))
reduce_sum = ReduceSumPrimitive()


def sub(x, y):
    return add(x, neg(y))


def div(x, y):
    return mul(x, reciprocal(y))


def transpose(x):
    return moveaxis(x, -1, -2)
