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

    Ax = np.dot(A, x_previous)
    difference = y - Ax

    return np.dot(A.T, difference)

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

def update_x(A, y, x, direction, c_inf, lambda_indices):
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
    gamma, new_index, lambda_indices, add = find_max_gamma(
        A, y, x, direction, c_inf, lambda_indices)

    x_next = gamma * direction
    return x_next, lambda_indices

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
    c = residual_vector(A, y, x)

    # initialise lambda = ||c||_inf
    c_inf = (np.linalg.norm(c, np.inf))

    # initialise vector to hold indices of maximum absolute values
    lambda_indices = np.array([False] * N, dtype=bool)

    ###### NB ONLY FIRST MAX VALUE RETURNED; IF LIKELY TO BE MULTIPLE NEED TO CHANGE THIS BIT ######
    lambda_indices[np.argmax(np.abs(c))] = True

    # evaluate the first direction vector
    A_gamma = A[:, lambda_indices]
    c_gamma = c[lambda_indices]
    invAtA = 1.0 / (np.linalg.norm(A_gamma) * np.linalg.norm(A_gamma))

    direction = np.zeros(N)
    direction[lambda_indices] = invAtA * helper.sign_vector(c_gamma)

    # update x
    gamma, new_index, lambda_indices, add = find_max_gamma(
        A, y, x, direction, c_inf, lambda_indices)

    x += gamma * direction
    
    effective_index = sum(lambda_indices[0:new_index])

    # evaluate homotopy path segments in iterations
    for i in range(0, N_iter):
        # update A_gamma and inverse_A_gamma
        invAtA = upd_inv.one_col_inv(A_gamma, invAtA, effective_index, A[:, new_index], add)
        A_gamma = A[:, lambda_indices]

        # update residual vector
        c = residual_vector(A, y, x)
        c_gamma = c[lambda_indices] # helper.subset_array(c, lambda_indices)

        # update direction vector
        direction.fill(0)
        direction[lambda_indices] = np.dot(invAtA, helper.sign_vector(c_gamma))

        # find lambda (i.e., infinity norm of residual vector)
        c_inf = (np.linalg.norm(c, np.inf))

        # print update
        print ("iteration {}\n  cinf={}\n  c={}\n  x={}\n".format(i, c_inf, c, x))

        # check if infinity norm of residual vector is within tolerance yet
        if np.linalg.norm(c, np.inf) < tolerance:
            break

        # update gamma and x
        gamma, new_index, lambda_indices, add = find_max_gamma(
            A, y, x, direction, c_inf, lambda_indices)
        
        x += gamma * direction

        # find where in A_gamma the new index fits in
        effective_index = sum(lambda_indices[0:new_index])

    return x

def solve(A, y, N_iter, tolerance):
    return homotopy_update(A, y, N_iter, tolerance)
