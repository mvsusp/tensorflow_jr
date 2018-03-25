# tensorflow_jr

An implementation of the TensorFlow framework for learning purposes

## Tensor

```python
class Tensor(object):
    def __init__(self, op, dtype, name):
        self.dtype = dtype
        self.op = op
        self.name = name
        self.shape = None
        self.value = None
```

The central unit of data in TensorFlow is the **tensor**. A tensor consists of a
set of primitive values shaped into an array of any number of dimensions.

TensorFlow uses numpy arrays to represent tensor **values**.

### Graph

```python
class Graph(object):
    def __init__(self):
        self.operations = []
        self.placeholders = []
        self.variables = []
```

A **computational graph** is a series of TensorFlow operations arranged into a
graph. The graph is composed of two types of objects.

  * __tf.Operation__ (or "ops"): The nodes of the graph.
    Operations describe calculations that consume and produce tensors.
    
```python
class Operation(object):
    def __init__(self, input_nodes, name):
        self.name = name
        self.input_nodes = input_nodes
        self.output_nodes = []

    def compute(self, args)
```    
    
  * __tf.Tensor__: The edges in the graph. These represent the values
    that will flow through the graph. Most TensorFlow functions return
    `tf.Tensors`.

Important: `tf.Tensors` do not have values, they are just handles to elements
in the computation graph.


Let's build a simple computational graph.

```python
x = tf.placeholder(dtype=tf.float32)

y = tf.placeholder(dtype=tf.float32)

W = tf.Variable(name='W', initial_value=np.array([1.]))

b = tf.constant(value=2.)

y_hat = x * W + b

```

```python
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
```

### TensorBoard
TensorFlow provides a utility called TensorBoard. One of TensorBoard's many
capabilities is visualizing a computation graph. You can easily do this with
a few simple commands.

First you save the computation graph to a TensorBoard summary file as
follows:

```
writer = tf.summary.FileWriter('.')
writer.add_graph(tf.get_default_graph())
```

This will produce an `event` file in the current directory with a name in the
following format:

```
events.out.tfevents.{timestamp}.{hostname}
```

Now, in a new terminal, launch TensorBoard with the following shell command:

```bsh
tensorboard --logdir .
```

TODO

Inspired by https://www.tensorflow.org/programmers_guide/low_level_intro
