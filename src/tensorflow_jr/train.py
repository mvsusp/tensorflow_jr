from src.tensorflow_jr import get_default_graph, UpdateGradient


class Optimizer(object):
    pass


class GradientDescentOptimizer(Optimizer):
    """Optimizer that implements the gradient descent algorithm.
    """

    def __init__(self, learning_rate):
        """Construct a new gradient descent optimizer.

    Args:
      learning_rate: A Tensor or a floating point value.  The learning
        rate to use.
    """
        self._learning_rate = learning_rate

    def minimize(self, loss, var_list):
        """Add operations to minimize `loss` by updating `var_list`.
        This method simply combines calls `compute_gradients()` and
        `apply_gradients()`. If you want to process the gradient before applying
        them call `compute_gradients()` and `apply_gradients()` explicitly instead
        of using this function.

        Args:
          loss: A `Tensor` containing the value to minimize.
          var_list: Optional list of `Variable` objects to update to minimize

      Returns:
          An Operation that updates the variables in `var_list`.
      """
        grads_and_vars = self.compute_gradients(loss, var_list)
        return self.apply_gradients(grads_and_vars)

    def compute_gradients(self, loss, var_list):
        return var_list

    def apply_gradients(self, grads_and_vars):
        return UpdateGradient(get_default_graph().variables)

