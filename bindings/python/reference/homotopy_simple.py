import math
import numpy as np

def Pfunc(lambda_t, alpha):
    if math.fabs(alpha) <= lambda_t:
        return 0
    else:
        if alpha > 0:
            return math.fabs(alpha) - lambda_t
        else:
            return -math.fabs(alpha) + lambda_t


def update_x(x, A, y, lambda_t):
    M, N = A.shape

    # initialise x_{t+1}
    x_next = np.zeros(N)

    # evaluate matrix products for P_lambda functional
    A_t = np.matrix.transpose(A)
    Ax = np.dot(A, x)
    Ax_m_y = np.zeros(M)
    for i in range(0, M):
        Ax_m_y[i] = Ax[i] - y[i]
    matProd = np.dot(A_t, Ax_m_y)

    argumand = np.zeros(N)
    for element in range(0, N):
        argumand[element] = x[element] - matProd[element]

    for i in range(0, N):
        x_next[i] = Pfunc(lambda_t, argumand[i])

    return x_next

def simple_homotopy(A, y, N_iter, gamma, s):
    M, N = A.shape

    # evaluate transpose of sensing matrix
    A_t = np.matrix.transpose(A)

    # initialise x to zero
    x = np.zeros(N)
    old_x = np.zeros(N)

    # initialise lambda
    lambda_t = np.linalg.norm(np.dot(A_t, y), np.inf)

    # run optimisation loop
    for t in range(0, N_iter):
        x = update_x(x, A, y, lambda_t)
        lambda_t = gamma * lambda_t

        print(str(lambda_t) + '\n')

        l1_norm = np.linalg.norm(x, 1)

        if l1_norm > (2.0 * s):
            return old_x

        old_x = x

    return x

