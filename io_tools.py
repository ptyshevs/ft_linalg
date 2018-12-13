from Matrix import Matrix


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


def output_result(x, solved=True, filename=None):
    """
    Check the result of gauss solve and output the result in proper format

    :param x:
    :param solved:
    :param filename:
    :return:
    """
    # if system has unique solution, we have Identity matrix in X
    f = None if filename is None else open(filename, "w+")
    if solved:
            print(" ".join([str(round(_, 5)) for _ in x]), file=f)
    else:
        # No solution or infinitely many solutions? Who cares?
        print(-1, file=f)
