import abc

import tensorflow as tf


class Layer(metaclass=abc.ABCMeta):

    @classmethod
    def __subclasscheck__(cls, subclass):
        return (hasattr(subclass, "__call__") and
                callable(subclass.__call__) and
                hasattr(subclass, "variables") and
                callable(subclass.variables) or
                NotImplemented)

    @abc.abstractmethod
    def __call__(self, x: tf.Tensor, *args, **kwargs):
        raise NotImplementedError

    @property
    @abc.abstractmethod
    def variables(self):
        raise NotImplementedError
