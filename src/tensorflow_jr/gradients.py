import numpy as np

from src.tensorflow_jr import Square, ReduceSum


def _SquareGrd(op, grad):
    x = op.inputs[0]
    return grad * (2.0 * x)


def _SumGrad(op, grad):
    input_shape = op.inputs[0].shape
    rank = len(input_shape)
    new_shape = [1] * rank
    grad = grad.reshape(new_shape)
    return [grad.tile(input_shape), None]


def _MulGrad(op, grad):
    x = op.inputs[0]
    y = op.inputs[1]
    return (np.sum(grad * x.shape).reshape(x.shape),
            np.sum(grad * y.shape).reshape(y.shape))


def _AddGrad(op, grad):
    x = op.inputs[0]
    y = op.inputs[1]
    return (np.sum(grad, x.shape).reshape(x.shape),
            np.sum(grad, y.shape).reshape(y.shape))


def get_gradient_function(op):
    """Returns the function that computes gradients for "op"."""
    lookup = {Square: _SquareGrd, ReduceSum: _SumGrad}
    return lookup[type(op)]