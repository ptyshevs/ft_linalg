import numpy as np
from Matrix import Matrix
from matrix_tools import eye
from gauss import SoleSolver, cond


def LU_factorization(A):
    nrow, ncol = A.shape if len(A.shape) == 2 else (A.shape, None)
    U = A[:, :]  # Make a copy (needed for np.array)
    one = eye(nrow)
    L = eye(nrow)

    for i in range(min((nrow, ncol))):
        if np.isclose(U[i, i], 0.0):  # find row with non-zero on the pivot place, swap
            swapped = False
            for j in range(i + 1, nrow):
                if not np.isclose(U[j, i], 0.0):
                    U[i, :], U[j, :] = U[j, :], U[i, :]
                    swapped = True
                    break
            if not swapped:  # either no or inf solutions
                break

        for j in range(i + 1, nrow):  # Remove corresponding coef. in other equations
            if not np.isclose(U[j, i], 0):
                k = U[j, i] / U[i, i]  # scale factor
                U[j, :] -= U[i, :] * k
                L[j, :] += one[i, :] * k
    return L, U


def LU_inv(A):
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
    L, U = LU_factorization(A)
    I = eye(nrow)
    Y = SoleSolver().solve(L, I)
    A_inv = SoleSolver().solve(U, Y)
    return A_inv


def LU_cond(A):
    return LU_inv(A).norm() * A.norm()


if __name__ == '__main__':
    # A = Matrix([[8, 2, 9],
    #             [4, 9, 4],
    #             [6, 7, 9]])
    # # A = Matrix([[2, 6, 2],
    # #             [-3, -8, 0],
    # #             [4, 9, 2]])
    # A_inv = LU_inv(A)
    # print(A_inv)
    # print("SHOULD BE EYE:")
    # print(A @ A_inv)
    A = Matrix([[-5, 0, 7, 0],
                [4, -24, 0, 1],
                [3, 12, -7, -23],
                [-2, 42, 37, -21]])
    b = Matrix([[144],
                [-170],
                [21],
                [-445]])
    print("cond(A):", LU_cond(A), "gauss cond(A):", cond(A))

