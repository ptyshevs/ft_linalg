import numpy as np
from Matrix import Matrix
from eigen import spectral_radius
from ft_linalg import eye, cut_diagonal, cut_lower_triangular
from gauss import gauss_inv


def simple_iterations_solve(A, b, x0=None, max_iter=100, eps=0.001, prepare=True):
    if x0 is not None:
        x = x0
    else:
        x = Matrix([[0] * b.shape[0]]).T
    if prepare:
        B = eye(A.shape[1]) - (2 * A) / A.norm()
        c = (b * 2) / A.norm()
    else:
        B = A
        c = b
    for i in range(max_iter):
        x_new = B @ x + c
        if (x_new - x).norm(2) < eps:
            break
        else:
            x = x_new
        # stopping condition
    print("# of iterations:", i)
    return x


def jacobi_solver(A, b, x0=None, max_iter=100, eps=0.001):
    nrow, ncol = A.shape
    if nrow != ncol:
        print("Cannot solve non-square SOLE!")
        return None
    D = cut_diagonal(A)
    R = A - D
    D_inv = eye(nrow) / D
    if x0 is None:
        x = Matrix([[0] * nrow]).T
    else:
        x = x0
    for i in range(max_iter):
        x_new = D_inv @ (b - R @ x)
        if (x - x_new).norm(2) < eps:
            break
        else:
            x = x_new
    print("# of iterations:", i)
    return x


# Successive Over Relaxation
def sor_solver(A, b, w=2, x0=None, max_iter=100, eps=10e-4):
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
        x_new = B @ x + c
        if (x - x_new).norm(2) < eps:
            break
        else:
            x = x_new
    print("# of iterations:", i)
    return x


def jacob_seidel_solver(A, b, x0=None, max_iter=100, tol=10e-4):
    """
    Same as sor_solver with w=1
    :param A:
    :param b:
    :param x0:
    :param max_iter:
    :return:
    """
    L = cut_lower_triangular(A, strict=False)
    U = A - L
    L_inv = gauss_inv(L)
    if x0 is not None:
        x = x0
    else:
        x = Matrix([[0] * b.shape[0]]).T
    for i in range(max_iter):
        x_new = L_inv @ (b - U @ x)
        if (x - x_new).norm(2) < tol:
            break
        else:
            x = x_new
    print("# of iterations:", i)
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

    # A = Matrix([[7.8, 5.3, 4.8],
    #             [3.3, 1.1, 1.8],
    #             [4.5, 3.3, 2.8]])
    # b = Matrix([[1.8],
    #             [2.3],
    #             [3.4]])

    #16 example
    A = Matrix([[3.8, 4.1, -2.3],
                [-2.1, 3.9, -5.8],
                [1.8, 1.1, -2.1]])
    b = Matrix([[4.8],
                [3.3],
                [5.8]])

    #Example from https://www.maa.org/press/periodicals/loci/joma/iterative-methods-for-solving-iaxi-ibi-gauss-seidel-method
    # solution x=(1, 2, -1).T
    A = Matrix([[4, -1, -1],
                [-2, 6, 1],
                [-1, 1, 7]])
    b = Matrix([[3],
                [9],
                [-6]])

    print("Matrix norm:", A.norm())
    print("Spectral raidus", spectral_radius(A))
    x = jacob_seidel_solver(A, b)
    print("solution:", x)
