# Copyright (c) 2025 by David Boetius
# Licensed under the MIT Licensed.
from .core import Value


class Placeholder:
    def __repr__(self):
        return "*"


flatteners = {
    tuple: lambda t: (t, None),
    list: lambda lst: (tuple(lst), None),
    dict: lambda d: (tuple(d.values()), d.keys()),
    type(None): lambda n: ((), None),
}

unflatteners = {
    tuple: lambda vals, _: tuple(vals),
    list: lambda vals, _: list(vals),
    dict: lambda vals, keys: {k: v for k, v in zip(keys, vals, strict=True)},
    type(None): lambda *_: None,
}


def is_value(arg):
    return isinstance(arg, Value)


def flatten(container, is_leaf=is_value):
    if is_leaf(container):
        return (container,), Placeholder()
    children, aux = flatteners[type(container)](container)
    if len(children) == 0:
        vals, subcontainers = (), ()
    else:
        vals, subcontainers = zip(*(flatten(child, is_leaf) for child in children))
        vals = sum(vals, start=())  # concat tuples
    structure = unflatteners[type(container)](subcontainers, aux)
    return vals, structure


def unflatten(structure, leaves):
    def _unflatten(treedef, leaf_iter):
        if isinstance(treedef, Placeholder):
            return next(leaf_iter)
        children, aux = flatteners[type(treedef)](treedef)
        subcontainers = [_unflatten(child, leaf_iter) for child in children]
        return unflatteners[type(treedef)](subcontainers, aux)

    leaf_iter = iter(leaves)
    try:
        container = _unflatten(structure, leaf_iter)
    except StopIteration:
        raise ValueError(f"Too few leaves provided for {structure}")
    if len(tuple(leaf_iter)) != 0:
        raise ValueError(f"Too many leaves provided for {structure}.")
    return container


def flat_expand_prefix(container, second, is_leaf=is_value):
    if is_leaf(container):
        return (second,)
    elif type(container) is not type(second):
        return (second,) * len(flatten(container, is_leaf)[0])
    else:
        children, aux = flatteners[type(container)](container)
        children2, aux2 = flatteners[type(container)](second)
        if aux != aux2 or len(children) != len(children2):
            raise ValueError(f"Prefix structure {second} does not match {container}")
        if len(children) == 0:
            return ()
        else:
            expanded = [flat_expand_prefix(c1, c2, is_leaf) for c1, c2 in zip(children, children2)]
            return sum(expanded, start=())


def map_structure(fn, container, *further_args, is_leaf=is_value):
    further_flat = [flat_expand_prefix(container, arg) for arg in further_args]
    flat, structure = flatten(container, is_leaf)
    results = [fn(arg, *further) for arg, *further in zip(flat, *further_flat, strict=True)]
    return unflatten(structure, results)
