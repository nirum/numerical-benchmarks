import numpy as np
import scipy

__all__ = ['cholesky', 'eig', 'svd']


def generate_symmetric(n):
    A = np.random.randn(n,n)
    return A.T.dot(A)


def cholesky(n):
    """
    Setup function for testing a scipy cholesky factorization

    """

    A = generate_symmetric(n)

    return scipy.linalg.cho_factor, (A,), {}


def eig(n):
    """
    Setup function for testing an eigendecomposition

    """

    # initialize a symmetric matrix
    C = generate_symmetric(n)

    return scipy.linalg.eig, (C,), {}


def svd(m,n):
    """
    Setup function for testing the singular value decomposition

    """

    # initialize
    A = np.random.randn(m,n)

    return scipy.linalg.svd, (A,), {}
