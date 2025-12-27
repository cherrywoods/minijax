# Copyright (c) 2025 by David Boetius
# Licensed under the MIT Licensed.
from typing import Any

from . import core
from .nested_containers import flatten, map_structure


def vmap(fn, in_axis: int | None | Any = 0, out_axis: int | None | Any = 0):
    def vmapped_fn(*args, **kwargs):
        vmapper = core.new_interpreter(VmapInterpreter, flatten(args)[0])
        vmap_vals = map_structure(lambda v, ax: Vmapped(vmapper, v, ax), args, in_axis)

        results = fn(*vmap_vals, **kwargs)

        results = map_structure(lambda vval, axis: vval.move_batch_axis(axis), results, out_axis)
        return map_structure(lambda vval: vval.base_value, results)

    return vmapped_fn


class Vmapped(core.Value):
    def __init__(self, interpreter, value, batch_axis: int | None):
        super().__init__(interpreter, value.shape)
        self.interpreter = interpreter
        self.base_value = value
        self.batch_axis = batch_axis

    def move_batch_axis(self, new_axis):
        if self.batch_axis is None or self.batch_axis == new_axis:
            return self
        new_base = core.moveaxis(self.base_value, self.batch_axis, new_axis)
        return Vmapped(self.interpreter, new_base, new_axis)


class VmapInterpreter(core.Interpreter):
    def __init__(self, level: int):
        super().__init__(level)

    def wrap(self, value):
        if isinstance(value, Vmapped):
            return value
        return Vmapped(self, value, None)

    def process(self, primitive, values, options):
        vvals = [vval.move_batch_axis(0) for vval in values]
        base_vals = [vval.base_value for vval in vvals]
        if primitive is core.dot:
            return vmap_dot(*vvals, **options)
        elif primitive in vmap_rules:
            result = vmap_rules[primitive](*base_vals, **options)
        else:
            result = primitive(*base_vals, **options)
        return Vmapped(self, result, batch_axis=0)


def vmap_dot(x: Vmapped, y: Vmapped):
    if len(y.shape) <= 2 and y.batch_axis is not None:
        y = y.move_batch_axis(1)
        out = core.dot(x.base_value, y.base_value)
        return Vmapped(x.interpreter, out, batch_axis=-1)
    out = core.dot(x.base_value, y.base_value)
    return Vmapped(x.interpreter, out, batch_axis=0)


def _shift(index):
    return index + 1 if index >= 0 else index


vmap_rules = {
    core.expand_dims: lambda x, axes: core.expand_dims(x, [_shift(ax) for ax in axes]),
    core.moveaxis: lambda x, **axes: core.moveaxis(x, **{k: _shift(v) for k, v in axes.items()}),
    core.reshape: lambda x, new_shape: core.reshape(x, x.shape[:1] + new_shape),
    core.reduce_sum: lambda x, axes: core.reduce_sum(x, [_shift(ax) for ax in axes]),
}
