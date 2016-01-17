"""
Speed tests for numerical computing in python

"""

from collections import defaultdict
import time_numpy
import time_scipy
import time_theano

import six
if six.PY2:
    from time import time as perf_counter
else:
    from time import perf_counter


def timeit(setup_func, *setup_args, **setup_kwargs):
    """
    Times a function with the given arguments

    """

    # initialize
    test_func, args, kwargs = setup_func(*setup_args, **setup_kwargs)

    # time the function execution
    t0 = perf_counter()
    test_func(*args, **kwargs)
    runtime = perf_counter() - t0

    return runtime


def run():

    tests = {
        'numpy': [
            (time_numpy.dot, 4000, 2000, 3000),
            (time_numpy.dot, 5000, 5000, 5000),
            (time_numpy.eig, 1000),
            (time_numpy.eig, 2000),
            (time_numpy.svd, 1000, 2000),
        ],

        'scipy': [
            (time_scipy.cholesky, 1000),
            (time_scipy.cholesky, 2000),
            (time_scipy.eig, 1000),
            (time_scipy.eig, 2000),
            (time_scipy.svd, 1000, 2000),
        ],

        'theano': [
            (time_theano.exp, 100000),
            (time_theano.exp, 1000000),
        ]
    }

    test_results = defaultdict(list)

    for module, functions in tests.items():
        print('Timing {}'.format(module))
        for func, *args in functions:
            print('\t{}\t{}'.format(func.__name__, args))
            key = '{}:{}'.format(module, func.__name__)
            test_results[key].append(timeit(func, *args))

    return test_results
