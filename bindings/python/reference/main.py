import os
import argparse
import numpy as np
import homotopy as ht
import homotopy_simple as shomt
import irls
import read_matrix_signal as read
import sparse_results as res
import suffixes as suffs

parser = argparse.ArgumentParser(description='Apply different sparse representation solvers.')

parser.add_argument(dest='filePrefix',
    help='input file prefix for input files containing matrix A (FILE_PREFIX_mat.dat) ' +
         'and signal (FILE_PREFIX_signal.dat)')

parser.add_argument('--sparse', dest='sparse',
    help='perform sparse decomposition using homotopy (--sparse h) or iterative ' +
         'reweighted least squares (--sparse i)' )

parser.add_argument('--algochoice', dest='choice',
    help='choice of algorithm (only used for homotopy); 1 for library solver, ' +
         '2 for continuous inverse updates')

args = parser.parse_args()

#read in matrix and signal
(A, b) = read.read_file(args.filePrefix)

#run sparse solver of choice
if args.sparse is not None:
    if os.path.isfile(args.filePrefix + suffs.fileMat + suffs.fileEnding) and os.path.isfile(
            args.filePrefix + suffs.fileSignal + suffs.fileEnding):
        if args.sparse == 'h':
            if args.choice == '2':
                x_sol = ht.homotopy(A, b, 10000, 1e-6, False)
                res.print_results(args.filePrefix, A, b, x_sol)
            else:
                x_sol = ht.homotopy(A, b, 10000, 1e-6, True)
                res.print_results(args.filePrefix, A, b, x_sol)

        elif args.sparse == 'hs':
            x_sol = shomt.simple_homotopy(A, b, 1000, 0.8, 0.2)
            res.print_results(args.filePrefix, A, b, x_sol)

        elif args.sparse == 'i':
            x_sol = irls.irls(A, b, 100, 3, 1e-7)
            res.print_results(args.filePrefix, A, b, x_sol)

        else:
            print('Error: Invalid sparse decomposition setting. '  +
                  'Please use homotopy (--sparse h) or iterative ' +
                  'reweighted least squares (--sparse i).\n')
    else:
        print('Error: The sparse representation could not be evaluated' +
              'because the file/s' + args.filePrefix + suffs.fileMat + suffs.fileEnding +
              'and/or' + args.filePrefix + suffs.fileSignal + suffs.fileEnding + 'do/does not exist.')



