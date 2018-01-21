import numpy as np
import numpy.ma as ma

def sign(x, tolerance=1e-7):
    """
    Function:  sign_vector
    --------------------
    computes the sign sequence of vector x (within some tolerance):
        sign_vector(x) = element-wise sign function on x

        x: input vector
        tolerance: the bounds around 0 to be considered unsigned

    returns: the sign sequence of vector x
    """
    z = np.copy(x)
    mask = ma.masked_inside(z, -tolerance, tolerance).mask
    z[mask] = 0
    
    return np.sign(z, out=z)
