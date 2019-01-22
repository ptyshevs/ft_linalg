from Matrix import Matrix
from matrix_tools import zeros, eye, is_close


def _qr_gram_schmidt(A):
    nrow, ncol = A.shape
    R = zeros(ncol)
    Q = zeros((nrow, ncol))
    for i in range(ncol):
        u = A[:, i] - sum([Q[:, j] * (A[:, i] @ Q[:, j].T)[0] for j in range(i)])
        e = u / u.norm(2)
        Q[:, i] = e
    for i in range(ncol):
        for j in range(ncol):
            if j >= i:
                R[i, j] = A[:, j] @ Q[:, i].T
    return Q, R


def qr(A, method='gram_schmidt'):
    """
    QR-factorization
    :param A:
    :param method:
    :return: Q (orthogonal matrix), R (upper-triangular rotation matrix)
    """
    if method == 'gram_schmidt':
        return _qr_gram_schmidt(A)


def lu(A):
    """
    LU-factorization using variation of Gaussian elimination
    :param A:
    :return: L, U
    """
    nrow, ncol = A.shape if len(A.shape) == 2 else (A.shape, None)
    U = A[:, :]  # Make a copy (needed for np.array)
    one = eye(nrow)
    L = eye(nrow)

    for i in range(min((nrow, ncol))):
        if is_close(U[i, i], 0.0):  # find row with non-zero on the pivot place, swap
            swapped = False
            for j in range(i + 1, nrow):
                if not is_close(U[j, i], 0.0):
                    U[i, :], U[j, :] = U[j, :], U[i, :]
                    swapped = True
                    break
            if not swapped:  # either no or inf solutions
                break

        for j in range(i + 1, nrow):  # Remove corresponding coef. in other equations
            if not is_close(U[j, i], 0):
                k = U[j, i] / U[i, i]  # scale factor
                U[j, :] -= U[i, :] * k
                L[j, :] += one[i, :] * k
    return L, U


if __name__ == '__main__':
    # example from http://www.cs.nthu.edu.tw/~cherung/teaching/2008cs3331/chap4%20example.pdf
    A = Matrix([[1, -1, 4],
                [1, 4, -2],
                [1, 4, 2],
                [1, -1, 0]])

    A = Matrix([[1, 1, 0],
                [1, 0, 1],
                [0, 1, 1]])
    Q, R = qr(A)
    print("RESULT:")
    print(Q @ R)
    print("Q.T @ Q = I")
    print(Q.T @ Q)
