# Copyright (c) 2025 by David Boetius
# Licensed under the MIT License.
import itertools
from dataclasses import dataclass
from functools import partial

from .core import Interpreter, Primitive, Value, new_interpreter
from .nested_containers import flatten, map_structure


def make_graph(fn, return_out_structure=False):
    def make_graph_fn(*args, **kwargs):
        make_cg = new_interpreter(MakeCG, flatten(args)[0])
        in_tracers = map_structure(partial(Tracer, make_cg), args)

        out_tracers = fn(*in_tracers, **kwargs)

        out_tracers, out_structure = flatten(out_tracers)
        graph = make_cg.graph(flatten(in_tracers)[0], out_tracers)
        if return_out_structure:
            return graph, out_structure
        else:
            return graph

    return make_graph_fn


_var_ids = itertools.count()


class Var:
    def __init__(self, shape):
        self._id = next(_var_ids)
        self.shape = shape
        self.is_const = False

    def __repr__(self) -> str:
        def letters(i):
            return chr(97 + i) if i < 26 else letters(i // 26 - 1) + chr(97 + (i % 26))

        return f"{letters(self._id)}{list(self.shape)}"


class Const:
    def __init__(self, value):
        self.value = value
        self.is_const = True

    def __repr__(self) -> str:
        return f"Const({self.value})"


@dataclass(frozen=True)
class Equation:
    primitive: Primitive
    inputs: tuple[Var | Const, ...]
    outvar: Var
    options: dict

    def __repr__(self) -> str:
        opts = ", ".join([f"{k}: {v}" for k, v in self.options.items()])
        opts = f"[{opts}]" if opts else ""
        repr = f"{self.outvar} = {self.primitive.name}{opts} "
        return repr + " ".join(map(str, self.inputs))


@dataclass(frozen=True)
class ComputeGraph:
    invars: tuple[Var, ...]
    outvars: tuple[Var, ...]
    equations: tuple[Equation, ...]

    def __repr__(self) -> str:
        repr = "input: " + " ".join(map(str, self.invars)) + "\n"
        repr += "\n".join([f"  {eqn}" for eqn in self.equations]) + "\n"
        return repr + "output: " + " ".join(map(str, self.outvars))


class Tracer(Value):
    def __init__(self, interpreter, value, const=False):
        super().__init__(interpreter, value.shape)
        self.value = value
        self.var = Const(value) if const else Var(value.shape)


class MakeCG(Interpreter[Tracer]):
    def __init__(self, level: int):
        super().__init__(level)
        self.equations = []

    def wrap(self, value):
        if isinstance(value, Tracer):
            return value
        return Tracer(self, value, const=True)

    def process(self, primitive, values, options):
        invals = [t.value for t in values]
        inputs = tuple(t.var for t in values)
        out = primitive(*invals, **options)
        if all(a.is_const for a in inputs):  # a for "Atom" = Var | Const
            out_tracer = Tracer(self, out, const=True)
        else:
            out_tracer = Tracer(self, out)
            eqn = Equation(primitive, inputs, out_tracer.var, options)
            self.equations.append(eqn)
        return out_tracer

    def graph(self, in_tracers, out_tracers):
        return ComputeGraph(
            tuple(t.var for t in in_tracers),
            tuple(t.var for t in out_tracers),
            tuple(self.equations),
        )
