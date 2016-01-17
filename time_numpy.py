import numpy as np

__all__ = ['dot', 'eig', 'svd']


def generate_symmetric(n):
    A = np.random.randn(n,n)
    return A.T.dot(A)


def dot(m,n,k):
    """
    Setup function for testing a numpy dot product

    """

    A = np.random.randn(m,n)
    B = np.random.randn(n,k)

    return np.dot, (A,B), {}


def eig(n):
    """
    Setup function for testing an eigendecomposition

    """

    # initialize a symmetric matrix
    C = generate_symmetric(n)

    return np.linalg.eig, (C,), {}


def svd(m,n):
    """
    Setup function for testing the singular value decomposition

    """

    # initialize
    A = np.random.randn(m,n)

    return np.linalg.svd, (A,), {}
