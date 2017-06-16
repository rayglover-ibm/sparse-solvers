import numpy as np

def one_col_inv(matA, current_inverse, pos_vCol, vCol, add):
    '''
    function: one_col_inv
    ---------------------
    updates the inverse current_inverse due to adding/removing column to/from X0

    matA:      original matrix A_gamma
    current_inverse: inverse of (matA^T  *  matA)
    pos_vCol:  the column index where the column is to be inserted or removed
    vCol:      column to be inserted or removed
    add:       boolean; true for insert column, false for remove column

    returns: the updated inverse, new_inverse
    '''

    # find dimension of original matrix
    M, N = matA.shape

    # initialise new inverse
    new_inverse = np.zeros((N+1, N+1))

    if add:
        # add column
        # compute the inverse as if adding a column to the end
        u1 = np.dot(matA.T, vCol)
        u2 = np.dot(current_inverse, u1)
        d  = 1.0/float(np.dot(vCol.T, vCol) - np.dot(u1.T, u2))
        u3 = d * u2

        F11inv = current_inverse + (d * np.outer(u2, u2.T))
        new_inverse[0:N, 0:N] = F11inv # [F11inv -u3 -u3' F22inv]
        new_inverse[0:N, N] = -u3
        new_inverse[N, 0:N] = -u3.T
        new_inverse[N, N] = d

        # permute to get the matrix corresponding to original X
        permute_order = np.hstack((np.arange(0, pos_vCol), N, np.arange(pos_vCol, N)))
        new_inverse = new_inverse[:, permute_order]
        new_inverse = new_inverse[permute_order, :]

    else:
        # remove column
        # re-size new_inverse
        new_inverse = np.zeros((N-1, N-1))

        # permute to bring the column at the end in X
        permute_order = np.hstack((np.arange(0, pos_vCol), np.arange(pos_vCol + 1, N), pos_vCol))
        current_inverse = current_inverse[permute_order, :]
        current_inverse = current_inverse[:, permute_order]

        # update the inverse by removing the last column
        F11inv = current_inverse[0:N-1, 0:N-1]
        d = current_inverse[N -1, N - 1]
        u3 = -1.0 * current_inverse[0:N - 1, N - 1]
        u2 = (1.0/d) * u3
        new_inverse = F11inv - (d * np.outer(u2, u2.T))

    return new_inverse
