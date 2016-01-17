from theano import function, config, shared
import theano.tensor as T
import numpy as np
import time

__all__ = ['exp']


def exp(n):
    """
    Setup function for calling the exponential function, using Theano

    """

    x = shared(np.random.randn(n), config.floatX)
    f = function([], T.exp(x))
    f.__name__ = 'exp'

    return f, (), {}
