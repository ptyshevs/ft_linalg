import numpy as np
from matrix_tools import eye
from io_tools import parse_assignment, split_input, output_result


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
        if nrow == ncol and self.is_singular(X):
            self.error = "singular matrix"
            return None
        self.n_solutions = 1
        return b

    def is_singular(self, A):
        """ Check if matrix is singular """
        if A.shape[0] != A.shape[1]:
            raise ValueError("Singular matrix is defined only for square matrices")
        if type(A) is np.ndarray:
            return not np.allclose(A, np.eye(A.shape[0]), rtol=0.01, atol=0.01)
        else:
            return not A.is_close(eye(A.shape[0]))


def gauss_inv(A):
    """
    Calculate inverse of matrix using Gaussian elimination on matrix A,
    expanded with identity matrix
    :param A:
    :return:
    """
    if A.shape[0] != A.shape[1]:
        raise ValueError("Inverse of non-square matrix")
    return SoleSolver().solve(A, eye(A.shape[0]))


def residual(A, x, b):
    """ Calculate residual vector """
    return A @ x - b


def cond(A, norm=1):
    """ Calculate condition number for matrix A """
    A_inv = gauss_inv(A)
    if A_inv is None:
        print("Singular matrix: Conditional number cannot be calculated")
        return -1
    return A.norm(norm) * A_inv.norm(norm)


if __name__ == '__main__':
    X, b = split_input(parse_assignment())
    b_solved = SoleSolver().solve(X, b.T)
    output_result(b_solved)
