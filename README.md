# minijax

![Jax to minijax](assets/minijax-gist.png "Minijax")

A mini deep learning framework à la [Jax](https://jax.readthedocs.io). 
Like Jax, minijax is based on function transformations. 
It implements `vmap`, `grad`, `jit` (well, about one third) in less than 500 lines of code.

```python
from minijax.core import relu
from minijax.eval import Array
from minijax.grad import grad
from minijax.jit import jit
from minijax.vmap import vmap

def model(x, params):
    for w, b in params:
        x = relu(w @ x + b)
    return x

# x = Array(...); define params
y = jit(grad(vmap(model, (0, None))))(x, params)
```

Each of `minijax.core`, `minijax.grad`, `minijax.vmap` and `minijax.jit` is less than 100 lines of code. 
A good place to get started is [`minijax/core.py`](https://github.com/cherrywoods/minijax/blob/main/src/minijax/core.py), the [`demo.ipynb`](https://mybinder.org/v2/gh/cherrywoods/minijax/binder?urlpath=%2Fdoc%2Ftree%2Fdemo.ipynb) notebook, or the [`train_mnist.ipynb`](https://mybinder.org/v2/gh/cherrywoods/minijax/binder?urlpath=%2Fdoc%2Ftree%2Ftrain_mnist.ipynb) notebooks.

 - [`demo.ipynb`](https://mybinder.org/v2/gh/cherrywoods/minijax/binder?urlpath=%2Fdoc%2Ftree%2Fdemo.ipynb) trains an small neural network classifier for a 2d dataset using stochastic gradient descent.
 - [`train_mnist.ipynb`](https://mybinder.org/v2/gh/cherrywoods/minijax/binder?urlpath=%2Fdoc%2Ftree%2Ftrain_mnist.ipynb) trains a multi-layer-perceptron on MNIST using Adam.
 - a little extra: [`derivatives.ipynb`](https://mybinder.org/v2/gh/cherrywoods/minijax/24ab76fedb2ee837381e62c90f398984b95476d9?urlpath=lab%2Ftree%2Fderivatives.ipynb) demonstrates computing higher-order derivatives using `minijax`.

To get started, clone this repository, run
```
pip install -e .[demos]
```
and have fun with the code!






## Acknowledgements
This repo is inspired by the awesome [micrograd](https://github.com/karpathy/micrograd/) repository that implements a jet smaller PyTorch-style deep learning framework. Unlike, micrograd, the purpose of minijax is to demonstrate composable function transformations. I also wanted something that scales at least to MNIST, so minijax uses [numpy](https://numpy.org) instead of pure Python scalars.

[Autodidax](https://docs.jax.dev/en/latest/autodidax.html) from the Jax docs taught me about the core ideas behind Jax. This repo is essentially Autodidax but in the micrograd format instead of a tutorial. I simplified some parts some more and tried to use less jargon.

I wrote minijax over the Christmas break in 2025 and thank my family for still having me around and doing 95% of the cooking.

---

## From minijax to Jax
This repo uses slightly different terminology than Jax: 
 - Nested containers (`minijax.nested_containers`) are PyTrees in Jax.
 - Compute graphs (`minijax.compute_grad`) are Jaxprs in Jax.

Besides lots of edge cases that are not handled in minijax, minijax also lacks most of Jax's jitting magic: minijax does not run code on GPUs or TPUs and can not split computation across multiple devices. Actually, minijax can only use a single CPU core. Besides that, minijax supports far fewer computation primitives (no convolutions, ...), has very unhelpful error messages, leaves broadcasting to numpy, has no dtypes (imagine that!), and and and...
