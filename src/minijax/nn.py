# Copyright (c) 2025 by David Boetius
# Licensed under the MIT Licensed.
import math

from .core import exp, log, reduce_sum, relu, reshape, square
from .eval import Array, zeros
from .nested_containers import flatten, map_structure
from .random import rand_uniform, split_rng_key


def linear(x, weight, bias):
    return x @ weight + bias


def mlp(x, params: list[dict[str, Array]]):
    x = reshape(x, (-1,))
    for layer_params in params[:-1]:
        x = linear(x, layer_params["weight"], layer_params["bias"])
        x = relu(x)
    return linear(x, params[-1]["weight"], params[-1]["bias"])


def softmax(x, axis: int):
    x_mean = reduce_sum(x, axis, keepaxes=True) / Array(x.shape[axis])
    exp_x = exp(x - x_mean)  # more numerically stable softmax
    return exp_x / reduce_sum(exp_x, axis, keepaxes=True)


# ======================================================================================================================


def reduce_mean(x):
    return reduce_sum(x) / Array(math.prod(x.shape))


def cross_entropy(y_pred, y_true):
    y_pred = softmax(y_pred, axis=-1)
    return -reduce_mean(reduce_sum(y_true * log(y_pred), axes=-1))


def weight_decay(params):
    param_norms = map_structure(lambda p: reduce_mean(square(p)), params)
    return sum(flatten(param_norms)[0], start=Array(0))


# ======================================================================================================================


def init_mlp(in_size, layer_sizes, rng_key):  # layer_sizes[-1] is output size
    in_sizes = (in_size,) + tuple(layer_sizes[:-1])
    rng_keys = split_rng_key(rng_key, len(layer_sizes))
    return [
        {"weight": kaiming_uniform((in_, out), in_, key), "bias": zeros((out,))}
        for in_, out, key in zip(in_sizes, layer_sizes, rng_keys)
    ]


def kaiming_uniform(shape, fan_in, rng_key):
    # kaiming_uniform initialization for ReLU
    bound = math.sqrt(2) * math.sqrt(3 / fan_in)
    return rand_uniform(shape, -bound, bound, rng_key=rng_key)
