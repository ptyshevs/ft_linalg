from Matrix import Matrix
from solvers import solve_jacobi, solve_jacobi_seidel


def test_jacobi_example_4x4():
    A = Matrix([[10, -1, 2, 0],
                [-1, 11, -1, 3],
                [2, -1, 10, -1],
                [0, 3, -1, 8]])
    b = Matrix([[6],
                [25],
                [-11],
                [15]])

    solution = solve_jacobi(A, b)
    assert solution == Matrix([[1],
                               [2],
                               [-1],
                               [1]])


def test_jacobi_seidel_3x3():
    #Example from https://www.maa.org/press/periodicals/loci/joma/iterative-methods-for-solving-iaxi-ibi-gauss-seidel-method
    # solution x=(1, 2, -1).T
    A = Matrix([[4, -1, -1],
                [-2, 6, 1],
                [-1, 1, 7]])
    b = Matrix([[3],
                [9],
                [-6]])
    solution = solve_jacobi_seidel(A, b)
    assert solution.is_close(Matrix([[1],
                                     [2],
                                     [-1]]))
