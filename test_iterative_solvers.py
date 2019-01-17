import numpy as np
from Matrix import Matrix
from IterativeSolve import jacobi_solver


def test_jacobi_example_4x4():
    A = Matrix([[10, -1, 2, 0],
                [-1, 11, -1, 3],
                [2, -1, 10, -1],
                [0, 3, -1, 8]])
    b = Matrix([[6],
                [25],
                [-11],
                [15]])

    solution = jacobi_solver(A, b)
    assert solution == Matrix([[1],
                               [2],
                               [-1],
                               [1]])
