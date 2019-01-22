from Matrix import Matrix
from matrix_tools import zeros, eye


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
    if method == 'gram_schmidt':
        return _qr_gram_schmidt(A)


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
