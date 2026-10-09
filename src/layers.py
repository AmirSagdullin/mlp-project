import numpy as np


def sigmoid(x):
    return 1.0 / (1.0 + np.exp(-np.clip(x, -500.0, 500.0)))


def sigmoid_derivative(a):
    return a * (1.0 - a)


def relu(x):
    return np.maximum(0.0, x)


def relu_derivative(a):
    return (a > 0.0).astype(a.dtype)


def softmax(x):
    x = x - np.max(x, axis=1, keepdims=True)
    e = np.exp(x)
    return e / np.sum(e, axis=1, keepdims=True)


ACTIVATIONS = {
    "sigmoid": (sigmoid, sigmoid_derivative),
    "relu": (relu, relu_derivative),
    "softmax": (softmax, None),
}

INITIALIZERS = {
    "heUniform": lambda fan_in: np.sqrt(2.0 / fan_in),
    "xavierUniform": lambda fan_in: np.sqrt(1.0 / fan_in),
    "random": lambda fan_in: 0.01,
}


class DenseLayer:
    def __init__(self, units, activation="sigmoid",
                 weights_initializer="heUniform"):
        if activation not in ACTIVATIONS:
            raise ValueError(f"Unknown activation: {activation}")
        if weights_initializer not in INITIALIZERS:
            raise ValueError(f"Unknown initializer: {weights_initializer}")

        self.units = units
        self.activation_name = activation
        self.weights_initializer = weights_initializer

        self.input_size = None
        self.W = None
        self.b = None
        self.x = None
        self.z = None
        self.a = None
        self.dW = None
        self.db = None

    def build(self, input_size, rng):
        self.input_size = input_size
        scale = INITIALIZERS[self.weights_initializer](input_size)
        self.W = rng.normal(0.0, scale, size=(input_size, self.units))
        self.b = np.zeros((1, self.units))

    def forward(self, x):
        self.x = x
        self.z = x @ self.W + self.b
        act = ACTIVATIONS[self.activation_name][0]
        self.a = act(self.z)
        return self.a

    def backward(self, grad_output):
        if self.activation_name == "softmax":
            grad_z = grad_output
        else:
            deriv = ACTIVATIONS[self.activation_name][1]
            grad_z = grad_output * deriv(self.a)
        self.dW = self.x.T @ grad_z
        self.db = np.sum(grad_z, axis=0, keepdims=True)
        return grad_z @ self.W.T

    def update(self, learning_rate):
        self.W -= learning_rate * self.dW
        self.b -= learning_rate * self.db

    def get_state(self):
        return {
            "units": self.units,
            "activation": self.activation_name,
            "weights_initializer": self.weights_initializer,
            "W": self.W,
            "b": self.b,
        }

    def set_state(self, state):
        self.units = state["units"]
        self.activation_name = state["activation"]
        self.weights_initializer = state["weights_initializer"]
        self.W = state["W"]
        self.b = state["b"]