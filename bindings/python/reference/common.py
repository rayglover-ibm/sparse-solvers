import numpy as np
import math

def sign_vector(x):
    """
    Function:  sign_vector
    --------------------
    computes the sign sequence of vector x (within some tolerance):
        sign_vector(x) = element-wise sign function on x

        x: input vector

    returns: the sign sequence of vector x
    """

    tolerance = 1e-7
    z = np.zeros(len(x))
    for i in range(0,len(x)):
        if math.fabs(x[i]) > tolerance:
            z[i] = np.sign(x[i])
        else:
            z[i] = 0
    return z

