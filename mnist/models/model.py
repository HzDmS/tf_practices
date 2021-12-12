from typing import List
from layers.layer import Layer

import tensorflow as tf


class Model:

    """
    Neural network model.
    """

    def __init__(self, layers: List[Layer]):
        """
        :param layers: a list of all layers of the neural network.
        """
        self.layers = layers

    def __call__(self, x: tf.Tensor, *args, **kwargs) -> tf.Tensor:
        """
        :param x: input tensor.
        :return: output tensor.
        """
        for layer in self.layers:
            x = layer(x)
        return x

    @property
    def variables(self) -> List[tf.Variable]:
        variables = []
        for layer in self.layers:
            variables.extend(layer.variables)
        return variables
