#iterative reweighted least squares algorithm for sparse decomposition

import numpy as np
import numpy.linalg as lin
import scipy.linalg as scilin

np.set_printoptions(precision=6, suppress=True)

def irls_newton(Q, R, b, w):
    C, upper = scilin.cho_factor(np.dot(Q.T, Q * w))
    s = scilin.cho_solve((C, upper), np.dot(Q.T, b))
    t = np.dot(Q, s)

    x = lin.solve(R, np.dot(Q.T, t))
    x /= lin.norm(x, ord=1)

    return x

def irls(A, b, n_iter, K, tol):
    """
    Function:  irls
    --------------------
    uses the iterative reweighted least squares method to solve the equation
        min||x||_1  subject to A x = y

        A: sensing matrix
        y: signal
        n_iter: maximum number of iterations
        K: sparsity budget

    returns: the sparse representation vector x
    """
    (M, N) = A.shape
    Q, R = lin.qr(A, mode='complete')

    # set up system
    epsi = 1.0      # initialise epsilon
    w = np.ones(N)  # initialise weights vector

    # run iterative procedure
    for _ in range(0, n_iter):
        # update x
        x = irls_newton(Q, R, b, w)
        print ("x={}".format(x))

        # sort r to get r-vector
        xsorted = np.sort(np.abs(x), kind='mergesort')[::-1] # sort in descending order

        # check if sparsity achieved yet
        if np.abs(xsorted[K + 1]) < tol:
            break

        # update epsilon if required
        epsi = min(epsi, xsorted[K + 1] / float(N))
        print ("epsi={}".format(epsi))

        # update weights
        w = 1.0 / np.sqrt(x * x + epsi * epsi)
        print ("w={}\n".format(w))

    # return final solution
    return x


b = np.asarray([.0, .5, .45, .05])
A = np.array([
    [.1, .2, .0, .2],
    [.4, .4, .8, .4],
    [.5, .3, .1, .4],
    [.0, .1, .1, .0]
])

x = irls(A, b, 20, 1, 0.01)
print("Ax=\n{0}\n".format(A.dot(x)))
