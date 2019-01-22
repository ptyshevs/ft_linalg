# This file contains API to various solvers implemented
from gauss import SoleSolver
from iterative_solvers import jacobi_solver, jacob_seidel_solver


def solve_gauss(A, b, *args, **kwargs):
    return SoleSolver(*args, **kwargs).solve(A, b)


def solve_jacobi(A, b, *args, **kwargs):
    return jacobi_solver(A, b, *args, **kwargs)


def solve_jacobi_seidel(A, b, *args, **kwargs):
    return jacob_seidel_solver(A, b, *args, **kwargs)


def solve(A, b, method='gauss', *args, **kwargs):
    """
    Single API to all methods implemented
    :param A: Matrix of coefficients
    :param b: Vector of free terms
    :param method: one of the following:
        gauss - Gaussian Elimination
    :return: Solution to Ax = b (x)
    """
    if method == 'gauss':
        return solve_gauss(A, b, *args, **kwargs)
    elif method == 'jacobi':
        return solve_jacobi(A, b, *args, **kwargs)
    elif method == 'jacobi_seidel':
        return solve_jacobi_seidel(A, b, *args, **kwargs)
    else:
        raise NotImplementedError(f"{method} is not implemented!")
