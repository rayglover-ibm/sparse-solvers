# Copyright 2017 International Business Machines Corporation
# 
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
# 
#     http://www.apache.org/licenses/LICENSE-2.0
# 
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.


import numpy as np
import numpy.linalg as lin
import scipy.linalg as scilin
import scipy.stats as stats

np.set_printoptions(suppress=True, linewidth=400)

def threshold(a, threshmin=None, newval=0):
    a = np.asarray(a).copy()
    mask = np.zeros(a.shape, dtype=bool)
    if threshmin is not None:
        mask |= (a < threshmin)
    a[mask] = newval
    return a

def irls_newton(Q, R, b, w):
    C = scilin.cholesky(np.dot(Q.T, Q * w), lower=False)
    s = scilin.cho_solve((C, False), np.dot(Q.T, b))
    t = np.dot(Q, s)
    x = scilin.solve_triangular(R, np.dot(Q.T, t), lower=False)
    return x

def solve(A, b, n_iter, tol, K=0):
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
    Q, R = lin.qr(A, mode='full')

    # initialise epsilon and weights
    w = np.ones(A.shape[1], dtype=A.dtype)
    epsi = 1.0
    p = 0.9

    # run iterative procedure
    for i in range(0, n_iter):
        # update x
        x = irls_newton(Q, R, b, w)
        reltol = np.max(x) * tol
        x = threshold(x, threshmin=reltol, newval=0)

        print ("iteration {}\n  w={}\n  x={}\n".format(i, w, x))

        # sort r to get r-vector
        xsorted = np.sort(np.abs(x), kind='mergesort')[::-1] # sort in descending order
        # check if sparsity achieved yet
        if np.abs(xsorted[K + 1]) <= reltol:
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
