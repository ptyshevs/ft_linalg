import numpy as np
from Matrix import Matrix
from ft_linalg import eye
from gauss import SoleSolver, cond





if __name__ == '__main__':
    # A = Matrix([[8, 2, 9],
    #             [4, 9, 4],
    #             [6, 7, 9]])
    # # A = Matrix([[2, 6, 2],
    # #             [-3, -8, 0],
    # #             [4, 9, 2]])
    # A_inv = LU_inv(A)
    # print(A_inv)
    # print("SHOULD BE EYE:")
    # print(A @ A_inv)
    A = Matrix([[-5, 0, 7, 0],
                [4, -24, 0, 1],
                [3, 12, -7, -23],
                [-2, 42, 37, -21]])
    b = Matrix([[144],
                [-170],
                [21],
                [-445]])
    print("cond(A):", LU_cond(A), "gauss cond(A):", cond(A))

