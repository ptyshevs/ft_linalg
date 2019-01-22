from decomposition import qr, lu, cholesky
from ft_linalg import eye
from Matrix import Matrix


def test_LU_example1_2x2():
    A = Matrix([[3, 1],
                [4, 2]])
    L, U = lu(A)
    assert L == Matrix([[1, 0],
                         [4/3, 1]])
    assert U == Matrix([[3, 1],
                         [0, 0.6666666666666667]])

    assert (L @ U) == A


def test_LU_example2_3x3_zero_row():
    A = Matrix([[1, 2, 3],
                [4, 5, 6],
                [7, 8, 9]])
    L, U = lu(A)
    assert (L @ U) == A


def test_LU_example3_3x3():
    A = Matrix([[2, 6, 2],
                [-3, -8, 0],
                [4, 9, 2]])
    L, U = lu(A)
    assert (L @ U) == A


def test_LU_example4_4x3():
    A = Matrix([[1, 0, 2, 3],
                [2, -1, 3, 6],
                [1, 4, 4, 0]])
    L, U = lu(A)
    assert (L @ U) == A
    L, U = lu(A.T)
    assert (L @ U) == A.T


def test_QR_example1_4x3():
    A = Matrix([[1, -1, 4],
                [1, 4, -2],
                [1, 4, 2],
                [1, -1, 0]])

    Q, R = qr(A)
    assert (Q @ R) == A
    assert (Q.T @ Q) == eye(A.shape[1])


def test_QR_example2_3x3():
    A = Matrix([[1, 1, 0],
                [1, 0, 1],
                [0, 1, 1]])

    Q, R = qr(A)
    assert (Q @ R).round() == A
    assert (Q.T @ Q).round() == eye(A.shape[1])


def test_Cholesky_3x3():
    A = Matrix([[4, 12, -16],
                [12, 37, -43],
                [-16, -43, 98]])

    L = cholesky(A)
    assert L.is_close(Matrix([[2, 0, 0],
                              [6, 1, 0],
                              [-8, 5, 3]]))
    assert (L @ L.T).is_close(A)
