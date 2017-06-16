import numpy as np
import math

def elementwise_absolute(in_vec):
    """
    Function:  elementwise_absolute
    --------------------
    returns the absolute values of each element of vector in_vec

        in_vec: input vector
        out_vec: output vector, containing the absolute values of the elements
                 of vector in_vec

    returns: the vector of absolute values of the elements of in_vec
    """
    N = len(in_vec)
    out_vec = np.zeros(N)
    for i in range(0,N):
        out_vec[i] = math.fabs(in_vec[i])
    return out_vec

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

def zero_mask(vec_in, index_set, N):
    """
    Function:  zero_mask
    --------------------
    constructs a vector containing zeros for all indices outside the index_set,
    and with values from vec_in according to the index_set

        vec_in: values to be placed in indices indicated by index_set
        index_set: set of integers indicating the elements we want to be non-zero
        N: length of entire output vector

    returns: the constructed vector vec_out
    """

    if N < max(index_set):
        print("Error: index out of bounds")

    # evaluate entire direction_vector by filling
    # indices not in lambda_indices with zero
    vec_out = np.zeros(N)
    iCount = 0

    for i in range(0, len(index_set)):
        if index_set[i]:
            vec_out[i] = vec_in[iCount]
            iCount = iCount + 1

    return vec_out

def subset_array(A, lambda_indices):
    """
    Function:  subset_matrix
    --------------------
    extracts the subset of columns in A numbered by integers in lambda_indices

        A: sensing matrix
        gamma_indices: set of integers indicating the columns we want to extract

    returns: the subsetted matrix A_{gamma}
    """
    n = A.ndim
    A_gamma = []

    if n == 1:
        for i_lambda in range(0, len(lambda_indices)):
            if lambda_indices[i_lambda]:
                A_gamma.append(A[i_lambda])
    elif n == 2:
        for i_lambda in range(0, len(lambda_indices)):
            if lambda_indices[i_lambda]:
                A_gamma.append(A[:,i_lambda])
    else:
        print("Error: subset_matrix not defined for more than two dimensions")

    A_gamma = np.array(A_gamma)
    A_gamma = np.matrix.transpose(A_gamma)
    return A_gamma

