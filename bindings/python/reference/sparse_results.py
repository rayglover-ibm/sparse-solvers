import numpy as np
import os
import suffixes as suffs

def print_results(filePrefix, A, y, x_sol):
    """
    Function:  print_results
    --------------------
    print to file filePrefix_sparse.dat: the sparse representation vector, x_sol
    print to screen: comparison of sparse representation vector, x_sol, to signal, b

        filePrefix: file prefix for output file
        A: sensing matrix
        y: signal
        x_sol: sparse representation vector

    returns: fail/success boolean
    """

    # print results to screen
    res = np.dot(A, x_sol)
    k = len(res)

    print('Sparse decomposition completed successfully; comparison ' +
          'of signal vector vs sparse representation: \n')

    for iE in range(0, k):
        print('signal: ' + str(y[iE]) + ' vs sparse representation: ' + str(res[iE]) + '\n')

    # print results to file
    if os.path.isfile(filePrefix + suffs.fileMat + suffs.fileEnding):
        x_sol.tofile(filePrefix + suffs.fileSparse + suffs.fileEnding, sep=suffs.separatorInput, format='%10.5f')
    else:
        print('Error: incorrect file path\n')
        return False

    # return true if success
    return True