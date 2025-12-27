# Copyright (c) 2025 by David Boetius
# Licensed under the MIT Licensed.
from . import core
from .compute_graph import make_graph
from .eval import Array, zeros
from .nested_containers import flatten, unflatten


def grad(fn):
    v_and_g_fn = value_and_grad(fn)
    return lambda *args, **kwargs: v_and_g_fn(*args, **kwargs)[1]


def value_and_grad(fn):
    def v_and_g_fn(*primals, **kwargs):
        return vjp(fn, return_primals=True)(primals, Array(1.0), **kwargs)

    return v_and_g_fn


def vjp(fn, return_primals=False):
    def vjp_fn(in_primals, out_tangents, **kwargs):
        cg = make_graph(fn)(*in_primals, **kwargs)

        in_primals, in_structure = flatten(in_primals)
        out_tangents, out_structure = flatten(out_tangents)

        primals = _forward(cg, in_primals)
        in_tangents = _backwards(cg, primals, out_tangents)

        in_tangents = unflatten(in_structure, in_tangents)
        if return_primals:
            out_primals = unflatten(out_structure, [primals[v] for v in cg.outvars])
            return out_primals, in_tangents
        else:
            return in_tangents

    return vjp_fn


def _forward(cg, primals):
    primals = {iv: p for iv, p in zip(cg.invars, primals, strict=True)}
    for eqn in cg.equations:
        args = [v.value if v.is_const else primals[v] for v in eqn.inputs]
        out = eqn.primitive(*args, **eqn.options)
        primals[eqn.outvar] = out
    return primals


def _backwards(cg, primals, out_tangents):
    tangents = {ov: t for ov, t in zip(cg.outvars, out_tangents)}

    def update(var, tangent):
        if var in tangents:
            tangents[var] = core.add(tangents[var], unbroadcast(tangent, primals[var]))
        elif not var.is_const:
            tangents[var] = unbroadcast(tangent, primals[var])

    for eqn in reversed(cg.equations):
        in_primals = [a.value if a.is_const else primals[a] for a in eqn.inputs]
        out_tangent = tangents[eqn.outvar] if eqn.outvar in tangents else Array(0.0)
        out_primal = primals[eqn.outvar]

        in_tangents = vjp_rules[eqn.primitive](out_tangent, out_primal, *in_primals, **eqn.options)

        in_tangents = (in_tangents,) if not isinstance(in_tangents, tuple) else in_tangents
        for v, t in zip(eqn.inputs, in_tangents, strict=True):
            update(v, t)

    return [tangents[iv] for iv in cg.invars]


def unbroadcast(tangent, primal):
    added_axes = [i for i in range(len(tangent.shape) - len(primal.shape))]
    return core.reduce_sum(tangent, tuple(added_axes))


def vjp_where(tangent, out, cond, true_val, false_val):
    zero = zeros(cond.shape)
    return (zero, core.where(cond, tangent, zero), core.where(cond, zero, tangent))


vjp_rules = {
    core.expand_dims: lambda t, _, x, axes: core.reduce_sum(t, axes),
    core.moveaxis: lambda t, _, __, source, destination: core.moveaxis(t, destination, source),
    core.reshape: lambda t, _, x, new_shape: core.reshape(t, x.shape),
    core.neg: lambda t, *_: -t,
    core.add: lambda t, *_: (t, t),
    core.reduce_sum: lambda t, _, x, axes: core.expand_dims(t, axes),
    core.dot: lambda t, _, x, y: (t @ core.transpose(y), core.transpose(x) @ t),
    core.mul: lambda t, _, x, y: (t * y, x * t),
    core.reciprocal: lambda t, _, x: -core.reciprocal(core.square(x)) * t,
    core.relu: lambda t, out, x: core.where(out, t, Array(0)),  # np.bool_(0) = False
    core.square: lambda t, _, x: t * Array(2) * x,
    core.sqrt: lambda t, _, x: t / (Array(2) * core.sqrt(x)),
    core.exp: lambda t, out, x: t * out,
    core.log: lambda t, _, x: t / x,
    core.where: vjp_where,
}
