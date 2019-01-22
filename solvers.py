# This file contains API to various solvers implemented
from gauss import SoleSolver


def solve_gauss(A, b):
    return SoleSolver().solve(A, b)


def solve(A, b, method='gauss'):
    """
    Single API to all methods implemented
    :param A: Matrix of coefficients
    :param b: Vector of free terms
    :param method: one of the following:
        gauss - Gaussian Elimination
    :return: Solution to Ax = b (x)
    """
    if method == 'gauss':
        return solve_gauss(A, b)
    else:
        raise NotImplementedError(f"{method} is not implemented!")
