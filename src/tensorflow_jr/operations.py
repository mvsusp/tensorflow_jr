import numpy as np

from src.tensorflow_jr.core import Operation, Tensor, Placeholder


class Square(Operation):
    def __init__(self, x):
        super(Square, self).__init__(input_nodes=[x], name='Square')

    @staticmethod
    def __compute__(x):
        return np.square(x)


class MatMult(Operation):
    def __init__(self, x, y):
        super(MatMult, self).__init__(input_nodes=[x, y], name='MatMult')

    @staticmethod
    def __compute__(x, y):
        return np.multiply(x, y)


def mat_mul(x, y):
    return MatMult(x, y)


class Add(Operation):
    def __init__(self, x, y):
        super(Add, self).__init__(input_nodes=[x, y], name='Add')

    @staticmethod
    def __compute__(x, y):
        return x + y


def add(x, y):
    return Add(x, y)


class Sub(Operation):
    def __init__(self, x, y):
        super(Sub, self).__init__(input_nodes=[x, y], name='Sub')

    @staticmethod
    def __compute__(x, y):
        return x - y


def sub(x, y):
    return Sub(x, y)


Tensor.__mul__ = mat_mul
Operation.__mul__ = mat_mul
Tensor.__add__ = add
Operation.__add__ = add
Tensor.__sub__ = sub
Operation.__sub__ = sub


def square(tensor):
    """Computes square of x element-wise.

    Args:
        tensor: A `Tensor`

    Returns:
        A `Tensor`. Has the same type as `x`.
    """
    return Square(tensor)


class ReduceSum(Operation):
    def __init__(self, x):
        super(ReduceSum, self).__init__(input_nodes=[x], name='ReduceSum')

    @staticmethod
    def __compute__(x):
        return np.sum(x)


def reduce_sum(tensor):
    return ReduceSum(tensor)


class NoOp(Operation):
    def __init__(self, input_nodes, name):
        super(NoOp, self).__init__(input_nodes, name)

    def __compute__(self):
        for variable in self.input_nodes:
            variable.value = variable.initial_value
        return None


class UpdateGradient(Operation):
    def __init__(self, x):
        super(UpdateGradient, self).__init__(input_nodes=x, name='UpdateGradient')

    def __compute__(self, *x):
        return None


def placeholder(dtype, name=None):
    return Placeholder(dtype=dtype, name=name)
