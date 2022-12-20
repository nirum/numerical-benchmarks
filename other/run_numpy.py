from timeit import repeat
import random
import numpy as np


if hasattr(np.__config__, "openblas_info"):
    BLAS = "openblas"
elif hasattr(np.__config__, "blas_mkl_info"):
    BLAS = "mkl"
else:
    BLAS = "unknown"


def mysum(numbers):
    """Compute sum of elements in x."""
    s = 0
    for n in numbers:
        s += n
    return s


f = open(f"python_{BLAS}.csv", "w")
print("language,blas,benchmark,time", file=f)

# builtin sum of 10**7 numbers
random.seed(1)
a = [random.uniform(0, 1) for _ in range(10**7)]
t = min(repeat(lambda: sum(a), number=50)) / 50
print(f"python,{BLAS},builtin_sum,{t}", file=f)

# custom sum of 10**7 numbers
t = min(repeat(lambda: mysum(a), number=10)) / 10
print(f"python,{BLAS},custom_sum,{t}", file=f)

# NumPy sum of 10**7 numbers
a = np.random.default_rng(1).uniform(size=(10**7))
t = min(repeat(lambda: np.sum(a), number=100)) / 100
print(f"python,{BLAS},numpy_sum,{t}", file=f)

# matrix multiplication
a = np.random.default_rng(1).uniform(size=(10000, 2000))
b = a.copy().T
t = min(repeat(lambda: a @ b, repeat=3, number=3)) / 3
print(f"python,{BLAS},mat_mult,{t}", file=f)

# Cholesky decomposition
a = np.random.default_rng(1).uniform(size=(3000, 3000))
b = a.copy().T
c = a @ b
t = min(repeat(lambda: np.linalg.cholesky(c), repeat=3, number=10)) / 10
print(f"python,{BLAS},cholesky,{t}", file=f)

# SVD
a = np.random.default_rng(1).uniform(size=(5000, 2000))
t = min(repeat(lambda: np.linalg.svd(a), repeat=3, number=3)) / 3
print(f"python,{BLAS},svd,{t}", file=f)

# determinant
a = np.random.default_rng(1).uniform(size=(500, 500))
t = min(repeat(lambda: np.linalg.det(a), number=1000)) / 1000
print(f"python,{BLAS},det,{t}", file=f)

# solve a linear equation system
a = np.random.default_rng(1).uniform(size=(2000, 2000))
b = np.arange(2000)
t = min(repeat(lambda: np.linalg.solve(a, b), number=50)) / 50
print(f"python,{BLAS},solve,{t}", file=f)

# least squares solution
a = np.random.default_rng(1).uniform(size=(2000, 1500))
b = np.arange(2000)
t = min(repeat(lambda: np.linalg.lstsq(a, b, rcond=None), number=5)) / 5
print(f"python,{BLAS},lstsqr,{t}", file=f)

f.close()
