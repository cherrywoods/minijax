# Copyright (c) 2025 by David Boetius
# Licensed under the MIT License.
from .compute_graph import ComputeGraph, make_graph
from .nested_containers import flatten, unflatten


def jit(fn):
    cache = {}

    def jit_fn(*args, **kwargs):
        nonlocal cache
        cache_key = tuple(kwargs.items())
        if cache_key not in cache:
            cg, out_structure = make_graph(fn, return_out_structure=True)(*args, **kwargs)
            cache[cache_key] = (prune_unused(cg), out_structure)  # in jax: compile
        return run_graph(*cache[cache_key], *args)  # in jax: lower

    return jit_fn


def prune_unused(compute_graph):  # Remove equations that do not contribute to the output
    used_vars = set(compute_graph.outvars)
    eqns = []
    for eqn in reversed(compute_graph.equations):
        if eqn.outvar in used_vars:
            used_vars.update([v for v in eqn.inputs if not v.is_const])
            eqns.insert(0, eqn)
    return ComputeGraph(compute_graph.invars, compute_graph.outvars, tuple(eqns))


def run_graph(compute_graph, out_structure, *args):
    args, _ = flatten(args)

    env = {v: a for v, a in zip(compute_graph.invars, args)}
    for eqn in compute_graph.equations:
        invals = [a.value if a.is_const else env[a] for a in eqn.inputs]
        env[eqn.outvar] = eqn.primitive(*invals, **eqn.options)
    outvals = [env[v] for v in compute_graph.outvars]
    return unflatten(out_structure, outvals)
