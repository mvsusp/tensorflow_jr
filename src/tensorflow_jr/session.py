import numpy as np

from src.tensorflow_jr import Constant, Operation, Placeholder, NoOp, get_default_graph, Variable, Tensor


class Session(object):
    def run(self, operations, feed_dict={}):
        try:
            iter(operations)
        except TypeError:
            operations = [operations]

        for op in operations:

            nodes_postorder = traverse_postorder(op)

            for node in nodes_postorder:
                print(node)

                if isinstance(node, Placeholder):
                    node.output = feed_dict[node]
                elif isinstance(node, Constant):
                    node.output = node.value
                elif isinstance(node, Variable):
                    node.output = node.value
                elif isinstance(node, Tensor):
                    node.output = node.value if node.value else node.op
                else:
                    # OPERATION
                    node.inputs = [n.output for n in node.input_nodes]

                    node.output = node.compute(node.inputs)

                if type(node.output) == list:
                    node.output = np.array(node.output)

        return [op.output for op in operations]


def traverse_postorder(operation):
    nodes_postorder = []

    def recurse(node):
        if isinstance(node, Operation):
            for input_node in node.input_nodes:
                recurse(input_node)
        nodes_postorder.append(node)

    recurse(operation)
    return nodes_postorder


def global_variables_initializer():
    op = NoOp(input_nodes=get_default_graph().variables, name='Init')
    return Tensor(op=op, dtype=None, name='Init')
