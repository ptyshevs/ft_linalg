from Matrix import Matrix
import collections


def eye(n):
    """
    Create nxn Identity matrix
    :param n: side
    :return:
    """
    return Matrix([[1 if i == j else 0 for i in range(n)] for j in range(n)])


def flipud(A):
    """
    Flip lower-upper triangular matrix
    :param A: Matrix
    :return:
    """
    return A[::-1]


def is_close(a, b, tol=1e-13):
    """ Python stores 15 digits after comma, thus this weird tolerance """
    return abs(a - b) < tol


def zeros(shape):
    """
    Create Matrix of size <shape>, filled with zeros.
    If shape is integer, create nxn zero matrix
    :param shape:
    :return:
    """
    if type(shape) is int:
        return Matrix([[0 for _ in range(shape)] for _ in range(shape)])
    elif isinstance(shape, collections.Sequence) and len(shape) == 2:
        return Matrix([[0 for _ in range(shape[1])] for _ in range(shape[0])])
    else:
        raise ValueError("Don't understand input shape:", shape)


def zeros_like(A):
    return zeros(A.shape)


def vec_to_diag(v):
    """
    Create a diagonal matrix from a vector
    :param v:
    :return:
    """
    n = len(v)
    A = zeros((n, n))
    for i in range(n):
        A[i, i] = v[i]
    return A


def argmax(A, axis=0):
    """
    Find index of maximum value in A
    :param A: Matrix
    :param axis: 0 for row index, 1 for column index, 2 for (row, col) tuple
    :return: index (-1 in case of error)
    """
    max_row, max_col, max_val = -1, -1, None
    if type(A) in (list, tuple):  # instead of failing miserably, find proper index
        for i, v in enumerate(A):
            if max_val is None:
                max_val = v
                max_row = i
            elif v > max_val:
                max_val = v
                max_row = i
        return max_row
    if A.shape == (0, 0):
        return max_row
    if A.shape[0] == 1 or A.shape[1] == 1:
        for i, val in enumerate(A):
            if max_val is None:
                max_val = val
                max_row = i
            elif val > max_val:
                max_val = val
                max_row = i
        return max_row
    for i, row in enumerate(A):
        for j, col in enumerate(row):
            if max_val is None:
                max_val = col
                max_row, max_col = i, j
            elif col > max_val:
                max_val = col
                max_row, max_col = i, j
    if axis == 0:
        return max_row
    elif axis == 1:
        return max_col
    else:
        return max_row, max_col


def cut_diagonal(A):
    nrow, ncol = A.shape
    D = eye(nrow)
    for i in range(nrow):
        D[i, i] = A[i, i]
    return D


def cut_lower_triangular(A, strict=True):
    nrow, ncol = A.shape
    X = A.copy()
    for i in range(nrow):
        for j in range(ncol):
            if j > i or (strict and j >= i):
                X[i, j] = 0
    return X


def to_file(A, filename):
    """ Save matrix coefficients to file """
    with open(filename, "w+") as f:
        print(A, file=f)


if __name__ == '__main__':
    v = Matrix([[1],
                [2],
                [3]])
    A = vec_to_diag([0, 2, 3, 1, 2, 33 , 21, 1 ,1 , 1, 3, 4])
    print(A)
