from Matrix import Matrix
from decomposition import qr
from matrix_tools import cut_diagonal


def power_iteration(A, x0=None, n_iter=100):
    """
    Power iteration finds eigenvector with the largest corresponding eigenvalue
    :param A:
    :param x0:
    :param n_iter:
    :return:
    """
    if x0 is None:
        x = Matrix([[1] for _ in range(A.shape[1])])
    else:
        x = x0
    for i in range(n_iter):
        v = A @ x
        x = v / v.norm(2)
    return x


def rayleigh_quotient(A, x, v0=None, n_iter=100):

    return (((A @ x).T @ x) / (x @ x.T))[0]


def eigenvalue(A, x):
    """

    :param A:
    :param x: eigvenvector of A
    :return:
    """
    x_scaled = A @ x
    return x_scaled[0] / x[0]


def spectral_radius(A):
    """
    Spectral radius of a matrix is it's largest absolute eigenvalue
    :param A:
    :return:
    """
    x = power_iteration(A)
    return abs(eigenvalue(A, x))


def qr_eigen(A, n_iter=10):
    """
    Search for all eigenvalues using QR algorithm
    :param A:
    :param n_iter:
    :return:
    """
    for i in range(n_iter):
        Q, R = qr(A)
        A = R @ Q
    return Matrix([[A[i, j] for j in range(A.shape[1]) if i == j] for i in range(A.shape[0])])


if __name__ == '__main__':
    # example from https://www.utdallas.edu/~herve/Abdi-EVD2007-pretty.pdf
    A = Matrix([[-2,4, 2],
                [-2, 1, 2],
                [4, 2, 5]])

    A = Matrix([[1, -3, 3],
                [3, -5, 3],
                [6, -6, 4]])

    # A = Matrix([[2, -12],
    #             [1, -5]])
    X = qr_eigen(A, 1000)
    print(X)
    print(X.shape)
