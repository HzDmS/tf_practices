import tensorflow as tf

from typing import List

from layers.layer import Layer


class DenseLayer(Layer):

    """
    A dense layer consists of a linear layer and a relu activation layer.
    """

    def __init__(self, name: str, shape: List[int]):
        """
        :param shape: shape of the weight.
        :param name: name of the dense layer.
        """
        self.w = tf.Variable(name=f"{name}:w", initial_value=tf.random.normal(shape), dtype=tf.float32)
        self.b = tf.Variable(name=f"{name}:w", initial_value=tf.zeros(shape[-1]), dtype=tf.float32)

    def __call__(self, x: tf.Tensor, *args, **kwargs) -> tf.Tensor:
        """
        :param x: input tense.
        :param args: args.
        :param kwargs: kwargs.
        :return: output of the dense layer.
        """
        x = tf.matmul(x, self.w) + self.b
        x = tf.nn.leaky_relu(x)
        return x

    @property
    def variables(self) -> List[tf.Variable]:
        return [self.w, self.b]
