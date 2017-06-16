import numpy as np
import suffixes as suffs

def normalise_columns(arrayA):
    '''
    Function: normalise_sensing_matrix
    ----------------------------------
    normalises the sensing matrix so that the columns of the sensing matrix add up to unity

    parameters:
        matA: input sensing matrix

    returns: the normalised sensing matrix
    '''

    rows_cols_tuple = arrayA.shape
    dims = len(rows_cols_tuple)

    if dims > 1:
        norm_factor = arrayA.sum(axis=0)
        for col in range(0, rows_cols_tuple[1]):
            for row in range(0, rows_cols_tuple[0]):
                arrayA[row, col] = arrayA[row, col] / norm_factor[col]
    else:
        norm_factor = arrayA.sum()
        for row in range(0, rows_cols_tuple[0]):
            arrayA[row] = arrayA[row] / norm_factor

    return arrayA

def read_file(filePrefix):
    """
    Function:  read_file
    --------------------
    reads the sensing matrix A from filePrefix_matrix.dat and the
    signal y from filePrefix_signal.dat

        A: sensing matrix
        y: signal vector

    returns: the sensing matrix A and the signal vector y
    """

    # sensing matrix
    M = 0   # number of sensing matrix rows
    N = 0   # number of sensing matrix columns
    A = []  # initialise list to hold sensing matrix rows

    with open(filePrefix + suffs.fileMat + suffs.fileEnding) as inF:
        for eachLine in inF:
            line = [x for x in eachLine.split(suffs.separatorInput)]
            M = M + 1
            A.append(list(map(float, line)))

    A = np.array(A) # convert list of lists to array representing sensing matrix

    # signal vector
    y = []           # initialise list to hold signal vector elements
    countLines = 0   # keep track that there is no more than one line in filePrefix_signal.dat

    with open(filePrefix + suffs.fileSignal + suffs.fileEnding) as inF:
        for eachLine in inF:
            line = [x for x in eachLine.split(suffs.separatorInput)]
            y = list(map(float, line))
            countLines = countLines + 1
            if countLines > 1:   #filePrefix_signal.dat should contain one line only
                print('Error: the signal input file can only contain one line of numbers')

    y = np.array(y)  # convert list to array representing signal vector

    # check dimensions agree
    if M != len(y):
        print('Error: the matrix and signal dimensions do not agree')

    # normalise sensing matrix and signal
    A = normalise_columns(A)
    y = normalise_columns(y)

    # return sensing matrix and signal
    return A, y