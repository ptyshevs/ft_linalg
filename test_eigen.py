from eigen import power_iteration, eigenvalue
from Matrix import Matrix
from matrix_tools import is_close


def test_example1_2x2():
    A = Matrix([[2, 3],
                [2, 1]])

    # A = Matrix([[2, -12],
    #             [1, -5]])
    x = power_iteration(A)
    eigv = eigenvalue(A, x)
    assert (A @ x).is_close(eigv * x)
    assert eigv == 4
    true_eigvec = Matrix([[3],
                   [2]])
    true_eigvec /= true_eigvec.norm(2)
    assert true_eigvec.is_close(x)

def test_example2_2x2():
    A = Matrix([[2, -12],
                [1, -5]])
    x = power_iteration(A)
    eigv = eigenvalue(A, x)
    assert (A @ x).is_close(eigv * x)
    assert is_close(eigv, -2, 1e06)
    true_eigvec = Matrix([[3],
                   [1]])
    true_eigvec /= true_eigvec.norm(2)
    assert true_eigvec.is_close(x)