#iterative reweighted least squares algorithm for sparse decomposition

import numpy as np
import math
import common as helpers

def update_x(A, y, w):
    """
    Function:  update_x
    --------------------
    updates x from previous iteration to the next

        A: sensing matrix
        y: signal vector
        w: weights

    returns: new updated approximation to the solution x
    """

    # find number of weights
    N = len(w)

    # construct diagonal matrix
    Dmat = np.zeros((N, N))
    for i in range(0,N):
        Dmat[i, i] = w[i]

    # find transpose of A
    A_t = np.matrix.transpose(A)

    # invert diagonal matrix
    Dmat_inv = np.linalg.inv(Dmat)

    # multiply inverse of D with transpose of A
    Dinv_At = np.dot(Dmat_inv, A_t)

    # invert ADA matrix
    ADA_inv = np.linalg.inv(np.dot(A, Dinv_At))

    # evaluate the transformation matrix
    pref_mat = np.dot(Dinv_At, ADA_inv)

    # evaluate x by applying the transformation matrix to y
    return np.dot(pref_mat, y)


def update_w(epsi, x):
    """
    Function:  update_w
    --------------------
    updates weights from previous iteration to the next

        epsi: current value of epsilon
        x: current best approximation to the solution x

    returns: new updated set of weights
    """

    N = len(x)
    w = np.zeros(N)
    for i in range(0, N):
        w[i] = 1.0 / math.sqrt((x[i] * x[i]) + (epsi * epsi))
    return w


def irls(A, b, N_iter, K, tol):
    """
    Function:  irls
    --------------------
    uses the iterative reweighted least squares method to solve the equation
        min||x||_1  subject to A x = y

        A: sensing matrix
        y: signal
        N_iter: maximum number of iterations
        K: sparsity budget

    returns: the sparse representation vector x
    """

    (M, N) = A.shape

    # set up system
    epsi = 1.0          # initialise epsilon
    w = np.ones(N)      # initialise weights vector
    x_sol = np.zeros(N) # initialise trial solution

    # run iterative procedure
    for iIter in range(0, N_iter):
        # update x
        x_sol = update_x(A, b, w)

        # sort r to get r-vector
        r_vec = np.sort(helpers.elementwise_absolute(x_sol), kind='mergesort')  # sort in ascending order
        # flip into descending order
        r_vec = r_vec[::-1]

        # check if sparsity achieved yet
        if math.fabs(r_vec[K+1] < tol):
            break

        # update epsilon if required
        testVal = r_vec[K + 1] / float(N)
        if testVal < epsi:
            epsi = testVal

        # update weights
        w = update_w(epsi, x_sol)

    # note number of iterations used
    print('sparse solver finished after ' + str(iIter + 1) + ' out of max ' + str(N_iter) + ' iterations')

    # return final solution
    return x_sol
