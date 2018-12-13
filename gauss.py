import numpy as np
from Matrix import Matrix
from matrix_tools import eye


class SoleSolver:
    def __init__(self):
        self.error = None
        self.n_solutions = None

    def is_close(self, a, b, tol=1e-13):
        """ Python stores 15 digits after comma, thus this weird tolerance """
        return abs(a - b) < tol

    def solve(self, A, b):
        """
        Solve SOLE using Gaussian elimination

        Available operations:
        1) Swapping two rows
        2) Multiplying a row by a nonzero number
        3) Adding a multiple of one row another row
        :param A: matrix of coefficients
        :param b: column vector of free terms
        :return: x column vector if SOLE is solvable
        :raises ValueError if SOLE is a singular matrix
        """
        nrow, ncol = A.shape if len(A.shape) == 2 else (A.shape, None)
        X = A[:, :]  # Make a copy (needed for np.array)
        b = b[:, :]  # Make a copy
        for i in range(min((nrow, ncol))):
            if self.is_close(X[i, i], 0.0):  # find row with non-zero on the pivot place, swap
                swapped = False
                for j in range(i + 1, nrow):
                    if not self.is_close(X[j, i], 0.0):
                        X[i, :], X[j, :] = X[j, :], X[i, :]
                        b[i, :], b[j, :] = b[j, :], b[i, :]
                        swapped = True
                        break
                if not swapped:  # either no or inf solutions
                    if self.is_close(X[i, i], 0) and self.is_close(b[i, 0], 0):
                        self.n_solutions = np.inf
                    elif self.is_close(b[i, 0], 0):
                        self.n_solutions = 0
            if not self.is_close(X[i, i], 1) and not self.is_close(X[i, i], 0.0):
                k = X[i, i]  # scale factor
                X[i, :] /= k  # scale coefficients
                b[i, :] /= k  # scale free term

            for j in range(i + 1, nrow):  # Remove corresponding coef. in other equations
                if not self.is_close(X[j, i], 0):
                    k = X[j, i]  # scale factor
                    X[j, :] -= X[i, :] * k
                    b[j, :] -= b[i, :] * k
            for j in range(i - 1, -1, -1):  # remove coef. above the main diagonal
                if not self.is_close(X[j, i], 0):
                    k = X[j, i]
                    X[j, :] -= X[i, :] * k
                    b[j, :] -= b[i, :] * k
        if self.is_singular(X):
            self.error = "singular matrix"
            return None
        self.n_solutions = 1
        return b


    def is_singular(self, A):
        """ Check if matrix is singular """
        if A.shape[0] != A.shape[1]:
            raise ValueError("Singular matrix is defined only for square matrices")
        if not A.is_close(eye(A.shape[0])):
            return True
        else:
            return False


def solve_gauss(A, b):
    """
    OLD

    """
    nrow, ncol = A.shape if len(A.shape) == 2 else (A.shape, None)
    X = A[:, :]  # Make a copy (needed for np.array)
    b = b[:, :]  # Make a copy
    for i in range(min((nrow, ncol))):
        if X[i, i] == 0:  # find row with non-zero on the pivot place, swap
            for j in range(i + 1, nrow):
                if X[j, i] != 0:
                    X[i, :], X[j, :] = X[j, :], X[i, :]
                    b[i, :], b[j, :] = b[j, :], b[i, :]
                    break
        if X[i, i] != 1 and X[i, i] != 0:
            k = X[i, i]  # scale factor
            X[i, :] /= k  # scale coefficients
            b[i, :] /= k  # scale free term

        for j in range(i + 1, nrow):  # Remove corresponding coef. in other equations
            if X[j, i] != 0:
                k = X[j, i]  # scale factor
                X[j, :] -= X[i, :] * k
                b[j, :] -= b[i, :] * k
        for j in range(i - 1, -1, -1):  # remove coef. above the main diagonal
            if X[j, i] != 0:
                k = X[j, i]
                X[j, :] -= X[i, :] * k
                b[j, :] -= b[i, :] * k
    if X.shape[0] == X.shape[1] and not X.is_close(eye(X.shape[0])):
        raise ValueError("Singular matrix")
    elif X.shape[0] > X.shape[1]:
        raise ValueError("Singular matrix")
    else:
        return b


def gauss_inv(A):
    """
    Calculate inverse of matrix using Gaussian elimination on matrix A,
    expanded with identity matrix
    :param A:
    :return:
    """
    if A.shape[0] != A.shape[1]:
        raise ValueError("Inverse of non-square matrix")
    A_inv = solve_gauss(A, np.eye(A.shape[0]))
    return A_inv


def residual(A, x, b):
    """ Calculate residual vector """
    return A @ x - b


def parse_assignment():
    """
    Parse format from the assignment (reading input from user, etc.)
    :return: Matrix
    """
    i = int(input())
    parsed = []
    for _ in range(i):
        parsed.append(list(map(float, input().split())))
    return Matrix(parsed)


def split_input(M):
    """
    Split input matrix into matrix of coefficients X and vector of constant
    terms b
    :param M:
    :return: X, b
    """
    b = M[:, -1]  # last column
    X = M[:, :-1]  # all except the last column
    return X, b


def condition_number(A, norm=2):
    """ Calculate condition number for matrix A """
    return A.norm(norm) * gauss_inv(A).norm(norm)


def output_result(x, solved=True, filename=None):
    """
    Check the result of gauss solve and output the result in proper format
    :param X:
    :return:
    """
    # if system has unique solution, we have Identity matrix in X
    f = None if filename is None else open(filename, "w+")
    if solved:
            print(" ".join([str(round(_, 5)) for _ in x]), file=f)
    else:
        # No solution or infinitely many solutions? Who cares?
        print(-1, file=f)


if __name__ == '__main__':
    X, b = split_input(parse_assignment())
    b_solved = solve_gauss(X, b.T)
    output_result(b_solved)
