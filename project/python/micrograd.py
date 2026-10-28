"""micrograd in pure Python: a faithful, Karpathy-style port.

Same model and training as cpp/micrograd.hpp. Deliberately NOT optimized:
it is the "plain Python" side of the speed comparison.

To match the C++ results to ~1e-15, a few things mirror the C++ exactly:
- A neuron computes 0 + w0*x0 + w1*x1 + ... + b (the notebook starts the sum at b).
- tanh uses math.tanh instead of the notebook's exp formula (std::tanh in C++).
- sigmoid uses the same overflow-safe formula as the C++.
"""

import math
import random


class Value:

    def __init__(self, data, _children=(), _op=''):
        self.data = data
        self.grad = 0.0
        self._backward = lambda: None
        self._prev = set(_children)
        self._op = _op

    def __repr__(self):
        return f"Value(data={self.data}, grad={self.grad})"

    def __add__(self, other):
        other = other if isinstance(other, Value) else Value(other)
        out = Value(self.data + other.data, (self, other), '+')

        def _backward():
            self.grad += 1.0 * out.grad
            other.grad += 1.0 * out.grad
        out._backward = _backward
        return out

    def __mul__(self, other):
        other = other if isinstance(other, Value) else Value(other)
        out = Value(self.data * other.data, (self, other), '*')

        def _backward():
            self.grad += other.data * out.grad
            other.grad += self.data * out.grad
        out._backward = _backward
        return out

    def __pow__(self, other):
        assert isinstance(other, (int, float)), "only supporting int/float powers for now"
        out = Value(self.data**other, (self,), f'**{other}')

        def _backward():
            self.grad += other * (self.data ** (other - 1)) * out.grad
        out._backward = _backward
        return out

    def __neg__(self):  # -self
        return self * -1

    def __sub__(self, other):  # self - other
        return self + (-other)

    def __truediv__(self, other):  # self / other
        return self * other**-1

    def __radd__(self, other):  # other + self
        return self + other

    def __rsub__(self, other):  # other - self
        return other + (-self)

    def __rmul__(self, other):  # other * self
        return self * other

    def __rtruediv__(self, other):  # other / self
        return other * self**-1

    def exp(self):
        x = self.data
        out = Value(math.exp(x), (self,), 'exp')

        def _backward():
            self.grad += out.data * out.grad
        out._backward = _backward
        return out

    def tanh(self):
        # math.tanh rather than the notebook's (e^2x - 1)/(e^2x + 1): same as std::tanh in C++
        t = math.tanh(self.data)
        out = Value(t, (self,), 'tanh')

        def _backward():
            self.grad += (1 - t**2) * out.grad
        out._backward = _backward
        return out

    def relu(self):
        out = Value(0.0 if self.data < 0 else self.data, (self,), 'relu')

        def _backward():
            self.grad += (self.data > 0) * out.grad
        out._backward = _backward
        return out

    def sigmoid(self):
        # overflow-safe: math.exp(1000) raises OverflowError, so never exp a positive number
        x = self.data
        if x >= 0:
            s = 1.0 / (1.0 + math.exp(-x))
        else:
            e = math.exp(x)
            s = e / (1.0 + e)
        out = Value(s, (self,), 'sigmoid')

        def _backward():
            self.grad += (s * (1 - s)) * out.grad
        out._backward = _backward
        return out

    def backward(self):
        topo = []
        visited = set()

        def build_topo(v):
            if v not in visited:
                visited.add(v)
                for child in v._prev:
                    build_topo(child)
                topo.append(v)
        build_topo(self)

        self.grad = 1.0
        for node in reversed(topo):
            node._backward()


def activate(x, activation):
    if activation == 'tanh':
        return x.tanh()
    if activation == 'relu':
        return x.relu()
    if activation == 'sigmoid':
        return x.sigmoid()
    if activation == 'linear':
        return x
    raise ValueError(f"unknown activation {activation!r}")


class Neuron:

    def __init__(self, nin, activation='tanh'):
        self.w = [Value(random.uniform(-1, 1)) for _ in range(nin)]
        self.b = Value(random.uniform(-1, 1))
        self.activation = activation

    def __call__(self, x):
        if len(x) != len(self.w):
            raise ValueError(f"Neuron: expected {len(self.w)} inputs, got {len(x)}")
        # sum(...) starts at 0 and the bias comes last: same order of additions as the C++
        act = sum(wi * xi for wi, xi in zip(self.w, x)) + self.b
        return activate(act, self.activation)

    def parameters(self):
        return self.w + [self.b]


class Layer:

    def __init__(self, nin, nout, activation='tanh'):
        self.neurons = [Neuron(nin, activation) for _ in range(nout)]

    def __call__(self, x):
        outs = [n(x) for n in self.neurons]
        return outs

    def parameters(self):
        return [p for n in self.neurons for p in n.parameters()]


class MLP:

    def __init__(self, nin, nouts, activation='tanh'):
        sz = [nin] + nouts
        # hidden layers use `activation`, the last layer is always linear
        self.layers = [Layer(sz[i], sz[i + 1], activation if i + 1 < len(nouts) else 'linear')
                       for i in range(len(nouts))]

    def __call__(self, x):
        for layer in self.layers:
            x = layer(x)
        return x

    def parameters(self):
        return [p for layer in self.layers for p in layer.parameters()]

    def set_parameters(self, values):
        params = self.parameters()
        if len(values) != len(params):
            raise ValueError(f"expected {len(params)} weights, got {len(values)}")
        for p, v in zip(params, values):
            p.data = float(v)
