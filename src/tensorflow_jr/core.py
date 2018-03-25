import numpy as np


class Graph(object):

    def __init__(self):
        self.operations = []
        self.placeholders = []
        self.variables = []


_default_graph = Graph()


def get_default_graph():
    return _default_graph


class Tensor(object):
    def __init__(self, op, dtype, name):
        self.dtype = dtype
        self.op = op
        self.name = name
        self.shape = None
        self.value = None

    def __str__(self):
        return "Tensor(\"{}\", shape={}, dtype={})".format(self.name, self.shape, self.dtype)


class Operation(object):
    def __init__(self, input_nodes, name):
        self.name = name
        self.input_nodes = input_nodes
        self.output_nodes = []

        for node in input_nodes:
            node.output_nodes.append(self)

        get_default_graph().operations.append(self)

    def __str__(self):
        return "Operation(\"{}\")".format(self.name)

    def compute(self, args):
        ret_val = self.__compute__(*args)
        print("Operation(\"{}\") {} = {}".format(self.name, args, ret_val))
        return ret_val


class Placeholder(Tensor):
    def __init__(self, dtype, name=None):
        super(Placeholder, self).__init__(name=name or 'Placeholder', op=None, dtype=dtype)
        self.dtype = dtype

        self.output_nodes = []
        get_default_graph().placeholders.append(self)


class Constant(object):
    def __init__(self, value, dtype):
        self.value = value
        self.dtype = dtype

        self.output_nodes = []
        get_default_graph().variables.append(self)


class Variable(object):
    def __init__(self, initial_value, name):
        self.name = name
        self.initial_value = initial_value
        self.value = initial_value
        self.output_nodes = []
        get_default_graph().variables.append(self)

    def __str__(self):
        return "Variable(\"{}\")".format(self.name)


def constant(value, dtype):
    return Constant(value, dtype)


def float32():
    return np.float32
