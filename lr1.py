# Pavel Tyshevskyi - Variant 2
from Matrix import Matrix
from gauss import SoleSolver, gauss_inv, cond


def A_task(s):
    A = Matrix([[2, 2, -1, 1],
                [-3, 0, 3, 0],
                [-1, 3, 3, 2],
                [1, 0, 0, 4]])
    b = Matrix([[3],
                [-9],
                [-7],
                [4]])
    return s.solve(A, b)


def G_task(s):
    A = Matrix([[-7, -6, -6, 6],
                [7, 6, 8, -13],
                [4, 17, -16, 10],
                [-4, 18, 19, 0]])
    b = Matrix([[144],
                [-170],
                [21],
                [-445]])
    return s.solve(A, b)

def K_task(s):
    A = Matrix([[5, 0, -7, 0],
                [-1, 6, 0, 1],
                [2, -6, -4, -5],
                [-6, -6, 15, 7]])
    b = Matrix([[-123],
                [60],
                [-108],
                [159]])
    return s.solve(A, b)


if __name__ == '__main__':
    s = SoleSolver()
    print(A_task(s))
    print()
    print(G_task(s))
    print()
    print(K_task(s))
    print(f"Error: {s.error}, number of solutions: {s.n_solutions}")
    A = Matrix([[5, 0, -7, 0],
                [-1, 6, 0, 1],
                [2, -6, -4, -5],
                [-6, -6, 15, 7]])
    gauss_inv(A)
