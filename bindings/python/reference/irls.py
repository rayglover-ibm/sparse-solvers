#iterative reweighted least squares algorithm for sparse decomposition

import numpy as np
import numpy.linalg as lin
import scipy.linalg as scilin
import scipy.stats as stats

np.set_printoptions(suppress=True)

def normZeroOne(x):
    return (np.min(x) - x) / -np.ptp(x)

def irls_newton(Q, R, b, w):
    C, upper = scilin.cho_factor(np.dot(Q.T, Q * w))
    s = scilin.cho_solve((C, upper), np.dot(Q.T, b))
    t = np.dot(Q, s)
    x = lin.solve(R, np.dot(Q.T, t))
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

    # initialise epsilon and weights
    w = np.ones(A.shape[1], dtype=A.dtype)
    epsi = 1.0
    p = 0.9

    # run iterative procedure
    for _ in range(0, n_iter):
        # update x
        x = irls_newton(Q, R, b, w)
        x = stats.threshold(x, threshmin=tol, newval=0)

        print ("w={}".format(w))
        print ("x={}\n".format(x))

        # sort r to get r-vector
        xsorted = np.sort(np.abs(x), kind='mergesort')[::-1] # sort in descending order

        # check if sparsity achieved yet
        if np.abs(xsorted[K + 1]) < tol:
            break

        # update epsilon if required
        epsi = min(epsi, xsorted[K + 1] / float(N))

        # update weights and normalize
        wnew = np.power(x * x + epsi, (p / 2) - 1)
        wnew /= np.sum(wnew)

        # stop if the weights aren't changing
        if abs(np.max(w - wnew)) < np.finfo(np.float32).eps:
            break

        w = wnew

    # return final solution
    return x / np.sum(x)

b = np.asarray(
    [0.27,  0.12,  0.25,  0.02,  0.27],
    dtype=np.float32
)
A = np.array([
    [0.25,  0.25,  0.29,  0.15,  0.14],
    [0.20,  0.15,  0.02,  0.16,  0.27],
    [0.15,  0.16,  0.29,  0.07,  0.09],
    [0.12,  0.25,  0.07,  0.25,  0.28],
    [0.20,  0.17,  0.29,  0.25,  0.14]
], dtype=np.float32)

x = irls(A, b, 10, 0, 0.01)

print("x={0}".format(x))
print("Ax={0}".format(A.dot(x)))
