# Pavel Tyshevskyi - Variant 2
from Matrix import Matrix
from gauss import solve_gauss


def A_task():
    A = Matrix([[2, 2, -1, 1],
                [-3, 0, 3, 0],
                [-1, 3, 3, 2],
                [1, 0, 0, 4]])
    b = Matrix([[3],
                [-9],
                [-7],
                [4]])
    return solve_gauss(A, b)


def G_task():
    A = Matrix([[-7, -6, -6, 6],
                [7, 6, 8, -13],
                [4, 17, -16, 10],
                [-4, 18, 19, 0]])
    b = Matrix([[144],
                [-170],
                [21],
                [-445]])
    return solve_gauss(A, b)

def K_task():
    A = Matrix([[5, 0, -7, 0],
                [-1, 6, 0, 1],
                [2, -6, -4, -5],
                [-6, -6, 15, 7]])
    b = Matrix([[-123],
                [60],
                [-108],
                [159]])
    return solve_gauss(A, b)


if __name__ == '__main__':
    print(A_task())
    print()
    print(G_task())
    print()
    print(K_task())
