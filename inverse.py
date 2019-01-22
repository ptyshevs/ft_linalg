from decomposition import lu
from solvers import solve
from ft_linalg import eye


def lu_inv(A):
    """
    Calculating matrix inverse using LU-factorization:

    A = LU
    A @ A_inv = I
    LU @ A_inv = I
    L (U @ A_inv) = I
    (U @ A_inv) = L_inv
    A_inv = L_inv @ U_inv
    :param A: matrix
    :return:
    """
    nrow, ncol = A.shape
    if nrow != ncol:
        print("Matrix is not invertible")
        return None
    L, U = lu(A)
    I = eye(nrow)
    Y = solve(L, I)
    A_inv = solve(U, Y)
    return A_inv


def lu_cond(A):
    return lu_inv(A).norm() * A.norm()
