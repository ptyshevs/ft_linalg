import numpy as np
from Matrix import Matrix
from matrix_tools import eye, cut_diagonal, cut_lower_triangular
from gauss import gauss_inv


def simple_iterations_solve(A, b, x0=None, max_iter=100, eps=0.001):
    if x0 is not None:
        x = x0
    else:
        x = Matrix([[0] * b.shape[0]]).T
    for i in range(max_iter):
        x = A @ x + b
        # stopping condition
    return x


def jacobi_solver(A, b, x0=None, max_iter=100, eps=0.001):
    nrow, ncol = A.shape
    if nrow != ncol:
        print("Cannot solve non-square SOLE!")
        return None
    D = eye(nrow)
    for i in range(nrow):
        D[i, i] = A[i, i]
    R = A - D
    D_inv = eye(nrow) / D
    if x0 is None:
        x = Matrix([[0] * nrow]).T
    else:
        x = x0
    for i in range(max_iter):
        x = D_inv @ (b - R @ x)
    return x

# Successive Over Relaxation
def sor_solver(A, b, w=2, x0=None, max_iter=100, eps=0.001):
    nrow, ncol = A.shape
    if nrow != ncol:
        print("Cannot solve non-square SOLE!")
        return None
    D = cut_diagonal(A)
    L = cut_lower_triangular(A)
    R = A - L - D

    B = gauss_inv(D + w * L) @ ((1 - w) * D - w * R)
    c = w * gauss_inv(D + w * L) @ b
    if x0 is not None:
        x = x0
    else:
        x = Matrix([[0] * b.shape[0]]).T
    for i in range(max_iter):
        x = B @ x + c
    return x


if __name__ == '__main__':
    # A = Matrix([[1.7, 2.8, 1.9],
    #             [2.1, 3.4, 1.8],
    #             [4.2, -1.7, 1.3]])
    # b = Matrix([[0.7],
    #             [1.1],
    #             [2.8]])
    # A = Matrix([[2.7, 3.3, 1.3],
    #             [3.5, -1.7, 2.8],
    #             [4.1, 5.8, -1.7]])
    # b = Matrix([[2.1],
    #             [1.7],
    #             [0.8]])
    # A = Matrix([[3.1, 2.8, 1.9],
    #             [1.9, 3.1, 2.1],
    #             [7.5, 3.8, 4.8]])

    A = Matrix([[7.8, 5.3, 4.8],
                [3.3, 1.1, 1.8],
                [4.5, 3.3, 2.8]])
    b = Matrix([[1.8],
                [2.3],
                [3.4]])

    x = sor_solver(A, b)
    print("solution:", x)