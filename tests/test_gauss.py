import numpy as np
from solvers import solve_gauss
from Matrix import Matrix
from matrix_tools import eye


def test_empty():
    """
    Test the equality of empty matrices.
    Doesn't work now because X[:, :] returns matrix of shape [1, 0], instead of [0, 0]
    :return:
    """
    a = Matrix([])
    try:
        x = solve_gauss(a, Matrix())
        assert False
    except ValueError as e:
        assert True

def test_single():
    """
    Test single element
    :return:
    """
    a = Matrix([[1]])
    x = solve_gauss(a, Matrix())
    assert x == []


def test_wiki():
    """
    Test example from wiki, using numpy arrays
    :return:
    """
    a = np.array([[2, 1, -1],
                  [-3, -1, 2],
                  [-2, 1, 2]], dtype=np.float64)
    b = np.array([[8],
                  [-11],
                  [-3]], np.float64)
    x = solve_gauss(a, b)
    print(x)
    assert np.all(x == np.array([[2],
                                 [3],
                                 [-1]]))


def test_with_mat():
    """
    4x4 SOLE using Matrix class
    :return:
    """
    X = Matrix([[1, 2, 1, 0],
                [5, 2, 4, 1],
                [2, 1, 2, 1],
                [3, 2, 2, 1]])
    b = Matrix([[4], [2], [0], [2]])
    x = solve_gauss(X, b)
    assert np.all(x.round(2) == Matrix([[0], [2], [0], [-2]]))


def test_non_solvable():
    """
    Simple 2x2 example with no solution
    :return:
    """
    X = Matrix([[7, 1], [14, 2]])
    b = Matrix([[-3], [1]])
    x = solve_gauss(X, b)
    assert x is None


def test_ex1():
    """
    Simple 2x2 example with unique solution
    :return:
    """
    X = Matrix([[3, -1],
                [-4, 2]])
    b = Matrix([[4], [2]])
    x = solve_gauss(X, b)
    assert x.round(2) == Matrix([[5], [11]])


def test_ex2():
    """
    Simple 2x2 example with unique solution
    :return:
    """
    X = Matrix([[3, 4],
                [-6, 3]])
    b = Matrix([[10], [-9]])
    x = solve_gauss(X, b)
    assert x.round(2) == Matrix([[2], [1]])


def test_ex3():
    """
    Simple 2x2 example with unique solution
    :return:
    """
    X = Matrix([[7, 4],
                [-2, 5]])
    b = Matrix([[-5], [26]])
    x = solve_gauss(X, b)
    assert x.round(2) == Matrix([[-3], [4]])


def test_ex4():
    """
    Simple 2x2 example with unique solution
    :return:
    """
    X = Matrix([[2, 4],
                [3, 5]])
    b = Matrix([[-12], [-16]])
    x = solve_gauss(X, b)
    assert x.round(2) == Matrix([[-2, -2]]).T


def test_ex5():
    """
    Simple 2x2 example with unique solution
    :return:
    """
    X = Matrix([[1, 2],
                [2, 4]])
    b = Matrix([[5], [2]])
    x = solve_gauss(X, b)
    assert x is None


def test_too_many_equations():
    """
    4 equations, but 3 variables, ie. too many constraints -> no solution
    :return:
    """
    X = Matrix([[0, 2, 1],
                [1, -2, -3],
                [-1, 1, 2],
                [0, 5, 2]])
    b = Matrix([[-8], [0], [3], [2]])
    x = solve_gauss(X, b)
    assert x.round(2) != Matrix([[-4, 5, -2]]).T


def test_too_many_variables():
    """
    3 equation, but 4 variables -> infinite number of solutions
    :return:
    """
    X = Matrix([[0, 2, 1, 0],
                [1, -2, -3, 5],
                [-1, 1, 2, 2]])
    b = Matrix([[-4, 5, -2]]).T
    x = solve_gauss(X, b)
    assert x.round(2) != Matrix([[-4, 5, -2]]).T


def test_ex6():
    """
    Simple examples from textbook
    :return:
    """
    X = Matrix([[1, 3, 4],
                [2, 7, 3],
                [2, 8, 6]])
    b = Matrix([[3], [-7], [-4]])
    x = solve_gauss(X, b)
    assert x.round(2) == Matrix([[4, -3, 2]]).T
    X = Matrix([[2, 8, -4],
                [2, 11, 5],
                [4, 18, 3]])
    b = Matrix([[0], [9], [11]])
    x = solve_gauss(X, b)
    assert x.round(2) == Matrix([[2, 0, 1]]).T
    X = Matrix([[0, 2, 6],
                [3, 9, 4],
                [1, 3, 5]])
    b = Matrix([[2], [7], [6]])
    x = solve_gauss(X, b)
    assert x.round(2) == Matrix([[7, -2, 1]]).T
    X = Matrix([[1, 3, 2, 5],
                [-1, 2, -2, 5],
                [2, 6, 4, 7],
                [0, 5, 2, 6]])
    b = Matrix([[11], [-6], [19], [5]])
    x = solve_gauss(X, b)
    assert x.round(2) == Matrix([[5, -1, 2, 1]]).T
