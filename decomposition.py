from Matrix import Matrix
import ft_linalg as ft_la


def _qr_gram_schmidt(A):
    nrow, ncol = A.shape
    R = ft_la.zeros(ncol)
    Q = ft_la.zeros((nrow, ncol))
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
    one = ft_la.eye(nrow)
    L = ft_la.eye(nrow)

    for i in range(min((nrow, ncol))):
        if ft_la.is_close(U[i, i], 0.0):  # find row with non-zero on the pivot place, swap
            swapped = False
            for j in range(i + 1, nrow):
                if not ft_la.is_close(U[j, i], 0.0):
                    U[i, :], U[j, :] = U[j, :], U[i, :]
                    swapped = True
                    break
            if not swapped:  # either no or inf solutions
                break

        for j in range(i + 1, nrow):  # Remove corresponding coef. in other equations
            if not ft_la.is_close(U[j, i], 0):
                k = U[j, i] / U[i, i]  # scale factor
                U[j, :] -= U[i, :] * k
                L[j, :] += one[i, :] * k
    return L, U


def cholesky(A):
    """
    Cholesky factorization for hermitian (symmetric) positive-definite matrix
    :param A:
    :return: L (lower triangular matrix such that L @ L.T = A
    """
    L = ft_la.zeros_like(A)
    nrow, ncol = A.shape
    for i in range(nrow):
        L[i, i] = (A[i, i] - sum([L[i, k] ** 2 for k in range(i)])) ** 0.5
        for j in range(i + 1, ncol):
            L[j, i] = (A[j, i] - sum([L[j, k] * L[i, k] for k in range(i)])) / L[i, i]
    return L


if __name__ == '__main__':
    # example from http://www.cs.nthu.edu.tw/~cherung/teaching/2008cs3331/chap4%20example.pdf
    A = Matrix([[4, 12, -16],
                [12, 37, -43],
                [-16, -43, 98]])
    L = cholesky(A)
    print("RESULT:")
    print(L)
    print("RESTORING MATRIX:")
    print(L @ L.T)
