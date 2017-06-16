import numpy as np
import math
import sys, os

import common as helper
import update_inverse_columns as upd_inv

def residual_vector(A, y, x_previous):
    """
    Function:  residual_vector
    --------------------
    computes the residual vector c:
        c_j = transpose(A) * (y  - (A * x_{j-1}))

        A: sensing matrix
        y: signal
        x_previous: best-estimate for x in previous iteration

    returns: the residual vector c
    """

    A_t = np.matrix.transpose(A)
    A_x = np.dot(A, x_previous)
    difference = np.zeros(len(y))
    for i in range(0, len(y)):
        difference[i] = y[i] - A_x[i]
    return np.dot(A_t, difference)

def find_max_gamma(A, y, x, dir_vec, c_inf, lambda_indices):
    """
    Function:  find_max_gamma
    --------------------
    computes the maximum extent of the jth homotopy path segment, gamma_j

        A: sensing matrix
        y: signal
        x: current best approximation to the solution
        dir_vec: direction vector for current homotopy path segment
        c_inf: infinite norm of residual vector
        lambda_indices: indices of support in current solution x

    returns: the updated set of lambda_indices, and
            the new gamma-value (i.e., the magnitude of new path segement)
    """

    # evaluate the eligible elements of transpose(A) * A * dir_vec
    M, N = A.shape
    AstarA = np.dot(np.matrix.transpose(A), A)
    AstarAd = np.dot(AstarA, dir_vec)             #evaluate transpose(A) * A * d

    # evaluate the eligible elements of the residual vector
    res_vec = residual_vector(A, y, x)

    # evaluate the competing lists of terms
    idx = np.inf
    current_minimal = np.inf

    # find the minimum term and its index
    for ell in range(0, N):
        previous_minimal = current_minimal
        if lambda_indices[ell]:
            minT = -x[ell] / dir_vec[ell]
            if minT > 0.0 and minT < current_minimal:
                current_minimal = minT
        else:
            left_test = math.fabs(1.0 - AstarAd[ell])
            right_test = math.fabs(1.0 + AstarAd[ell])
            if left_test > 0:
                leftT = (c_inf - res_vec[ell]) / (1.0 - AstarAd[ell])
                if leftT > 0.0 and leftT < current_minimal:
                    current_minimal = leftT
            if right_test > 0:
                rightT = (c_inf + res_vec[ell]) / (1.0 + AstarAd[ell])
                if rightT > 0.0 and rightT < current_minimal:
                    current_minimal = rightT

        if previous_minimal > current_minimal:
            idx = ell

    # return the min bound on gamma and the corresponding index of x
    if lambda_indices[idx]:
        lambda_indices[idx] = False
        return current_minimal, idx, lambda_indices, False
    else:
        lambda_indices[idx] = True
        return current_minimal, idx, lambda_indices, True

def update_direction(A, res_vec, lambda_indices):
    """
    Function:  update_direction
    --------------------
    computes the direction vector d by solving the system of equations
        transpose(A_{gamma}) * A_{gamma} * d_{gamma} = sign_vector(c)

        A: sensing matrix
        y: signal
        x: current best approximation to solution
        lambda_indices: indices of support in current solution x

    returns: the new direction vector d
    """
    M, N = A.shape

    # evaluate the subsamples matrices for solving the equation
    A_gamma = helper.subset_array(A, lambda_indices)
    matProd = np.dot(np.matrix.transpose(A_gamma), A_gamma)
    res_vec_gamma = helper.subset_array(res_vec, lambda_indices)

    # solve equation
    dSol = np.linalg.solve(matProd, helper.sign_vector(res_vec_gamma))

    # evaluate entire direction_vector by filling indices not in lambda_indices with zero
    direction_vector = helper.zero_mask(dSol, lambda_indices, N)

    return direction_vector

def update_x(A, y, x, dir_vec, c_inf, lambda_indices):
    """
    Function:  update_x
    --------------------
    computes the new approximation to the solution x by adding another
    homotopy path segment to the previous approximation of x
        x^(j) = x^(j-1) + gamma_j d^(j)

        A: sensing matrix
        y: signal
        x: current best approximation to solution
        c_inf: the infinite norm of the residual vector
        lambda_indices: indices of support in current solution x

    returns: the new approximation to the solution x^(j) and the updated support lambda_indices
    """

    N = len(x)
    x_next = np.zeros(N)

    gamma, new_index, lambda_indices, add = find_max_gamma(A, y, x, dir_vec, c_inf, lambda_indices)

    for i in range(0, N):
        x_next[i] = x[i] + (gamma * dir_vec[i])

    return x_next, lambda_indices

def homotopy_absolute(A, y, N_iter, tolerance):
    """
    Function:  homotopy
    --------------------
    uses the homotopy method to solve the equation
        min||x||_1  subject to A x = y

        A: sensing matrix
        y: signal
        N_iter: maximum number of iterations
        tolerance: sparsity budget

    This function solves the equation
        transpose(A) * A * d = sign(c)
    using an Ax = b solver library.

    returns: the sparse representation vector x
    """

    # find dimensions of problem
    M, N = A.shape

    # initialise x to a vector of zeros
    x = np.zeros(N)

    # initialise residual vector
    c_vec = residual_vector(A, y, x)

    # initialise lambda = ||c_vec||_inf
    c_inf = (np.linalg.norm(c_vec, np.inf))

    # initialise vector to hold indices of maximum absolute values
    lambda_indices = [False] * N

    ###### NB ONLY FIRST MAX VALUE RETURNED; IF LIKELY TO BE MULTIPLE NEED TO CHANGE THIS BIT ######
    first_index = np.argmax(helper.elementwise_absolute(c_vec))
    lambda_indices[first_index] = True

    # evaluate the first direction vector
    A_gamma = helper.subset_array(A, lambda_indices)
    c_vec_gamma = helper.subset_array(c_vec, lambda_indices)
    prefactor = 1.0 / (np.linalg.norm(A_gamma) * np.linalg.norm(A_gamma))
    subsample_direction_vector = prefactor * helper.sign_vector(c_vec_gamma)
    direction_vec = helper.zero_mask(subsample_direction_vector, lambda_indices, N)

    # update x
    x, lambda_indices = update_x(A, y, x, direction_vec, c_inf, lambda_indices)

    # evaluate homotopy path segments in iterations
    for iteration in range(0, N_iter):
        #update residual vector
        c_vec = residual_vector(A, y, x)

        # update direction vector
        direction_vec = update_direction(A, c_vec, lambda_indices)

        # find lambda (i.e., infinite norm of residual vector)
        c_inf = (np.linalg.norm(c_vec, np.inf))

        # check if infinity norm of residual vector is within tolerance yet
        if np.linalg.norm(c_vec, np.inf) < tolerance:
            break

        # update gamma and x
        x, lambda_indices = update_x(A, y, x, direction_vec, c_inf, lambda_indices)

        # print update
        print('iteration ' + str(iteration + 1) + ' yields error ' + str(c_inf) + '\n')

    print('sparse solver finished after ' + str(iteration) + ' out of max ' + str(N_iter) + ' iterations')

    return x

def homotopy_update(A, y, N_iter, tolerance):
    """
    Function:  homotopy_update
    --------------------
    uses the homotopy method to solve the equation
        min||x||_1  subject to A x = y

        A: sensing matrix
        y: signal
        N_iter: maximum number of iterations
        tolerance: sparsity budget

    This function solves the equation
        transpose(A) * A * d = sign(c)
    by continuously updating (transpose(A) * A)^(-1).

    returns: the sparse representation vector x
    """

    M, N = A.shape

    # initialise x to a vector of zeros
    x = np.zeros(N)

    # initialise residual vector
    c_vec = residual_vector(A, y, x)

    # initialise lambda = ||c_vec||_inf
    c_inf = (np.linalg.norm(c_vec, np.inf))

    # initialise vector to hold indices of maximum absolute values
    lambda_indices = [False] * N

    ###### NB ONLY FIRST MAX VALUE RETURNED; IF LIKELY TO BE MULTIPLE NEED TO CHANGE THIS BIT ######
    first_index = np.argmax(helper.elementwise_absolute(c_vec))
    lambda_indices[first_index] = True

    # evaluate the first direction vector
    A_gamma = helper.subset_array(A, lambda_indices)
    c_vec_gamma = helper.subset_array(c_vec, lambda_indices)
    invAtA = 1.0 / (np.linalg.norm(A_gamma) * np.linalg.norm(A_gamma))
    subsample_direction_vector = invAtA * helper.sign_vector(c_vec_gamma)
    direction_vec = helper.zero_mask(subsample_direction_vector, lambda_indices, N)

    # update x
    gamma, new_index, lambda_indices, add = find_max_gamma(A, y, x, direction_vec, c_inf, lambda_indices)
    for i in range(0, N):
        x[i] = x[i] + (gamma * direction_vec[i])
    effective_index =  sum(lambda_indices[0:new_index])

    # evaluate homotopy path segments in iterations
    for iteration in range(0, N_iter):
        # update A_gamma and inverse_A_gamma
        invAtA = upd_inv.one_col_inv(A_gamma, invAtA, effective_index, A[:,new_index], add)
        A_gamma = helper.subset_array(A, lambda_indices)

        # update residual vector
        c_vec = residual_vector(A, y, x)
        c_vec_gamma = helper.subset_array(c_vec, lambda_indices)

        # update direction vector
        direction_vec = np.dot(invAtA, helper.sign_vector(c_vec_gamma))
        direction_vec = helper.zero_mask(direction_vec, lambda_indices, N)

        # find lambda (i.e., infinity norm of residual vector)
        c_inf = (np.linalg.norm(c_vec, np.inf))

        # check if infinity norm of residual vector is within tolerance yet
        if np.linalg.norm(c_vec, np.inf) < tolerance:
            break

        # update gamma and x
        gamma, new_index, lambda_indices, add = find_max_gamma(A, y, x, direction_vec, c_inf, lambda_indices)
        for i in range(0, N):
            x[i] = x[i] + (gamma * direction_vec[i])

        # find where in A_gamma the new index fits in
        effective_index = sum(lambda_indices[0:new_index])

        # print update
        print('iteration ' + str(iteration + 1) + ' yields error ' + str(c_inf) + '\n')

    print('sparse solver finished after ' + str(iteration) + ' out of max ' + str(N_iter) + ' iterations')

    return x

def homotopy(A, y, N_iter, tolerance, algorithm_choice):
    x = False
    if algorithm_choice:
        x = homotopy_absolute(A, y, N_iter, tolerance)
    else:
        x = homotopy_update(A, y, N_iter, tolerance)
    return x