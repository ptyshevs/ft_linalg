import numpy as np

from LUFactorization import LU_factorization
from Matrix import Matrix


def test_example1_2x2():
    A = Matrix([[3, 1],
                [4, 2]])
    L, U = LU_factorization(A)
    assert L == Matrix([[1, 0],
                         [4/3, 1]])
    assert U == Matrix([[3, 1],
                         [0, 0.6666666666666667]])

    assert (L @ U) == A


def test_example2_3x3_zero_row():
    A = Matrix([[1, 2, 3],
                [4, 5, 6],
                [7, 8, 9]])
    L, U = LU_factorization(A)
    assert (L @ U) == A


def test_example3_3x3():
    A = Matrix([[2, 6, 2],
                [-3, -8, 0],
                [4, 9, 2]])
    L, U = LU_factorization(A)
    assert (L @ U) == A


def test_example4_4x3():
    A = Matrix([[1, 0, 2, 3],
                [2, -1, 3, 6],
                [1, 4, 4, 0]])
    L, U = LU_factorization(A)
    assert (L @ U) == A
    L, U = LU_factorization(A.T)
    assert (L @ U) == A.T