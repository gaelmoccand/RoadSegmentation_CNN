# -*- coding: utf-8 -*-
"""Useful methods for project 1"""
import numpy as np

# ***************************************************
# COSTS
# ***************************************************
def calculate_mse(e):
    """Calculate the mse from the input vector"""
    N = len(e)
    mse = e.dot(e)/(2*N)
    return mse

# ***************************************************
# GRADIENT DESCENT
# ***************************************************
def compute_gradient(y, tx, w):
    """Compute the gradient."""
    error = y - tx.dot(w)
    N = len(error)
    grad = -tx.T.dot(error)/N
    
    return grad, error


def gradient_descent(y, tx, initial_w, max_iters, gamma):
    """Gradient descent algorithm."""
    # Define parameters to store w and loss
    ws = [initial_w]
    losses = []
    w = initial_w
    for n_iter in range(max_iters):
        # Compute gradient and loss
        grad, error = compute_gradient(y, tx, w)
        loss = compute_mse(error)
        # Update w by gradient
        w = w - gamma*grad
        # store w and loss
        ws.append(w)
        losses.append(loss)
        print("Gradient Descent({bi}/{ti}): loss={l}, w0={w0}, w1={w1}".format(
              bi=n_iter, ti=max_iters - 1, l=loss, w0=w[0], w1=w[1]))

    return losses, ws

# ***************************************************
# LEAST SQUARES
# ***************************************************	
def least_squares_GD(y, tx, initial_w, max_iters, gamma):
    """Linear regression using gradient descent"""
    raise NotImplementedError
    

def least_squares_SGD(y, tx, initial_w, max_iters, gamma):
    """Linear regression using stochastic gradient descent"""
    raise NotImplementedError
    

def least_squares(y, tx):
    """Least squares regression using normal equations"""
    # We want to solve the linear system Aw = b...
    # ...with A being the Gram Matrix...
    A = tx.T.dot(tx)
    # ... and b being the transpose of tx times y    """Least squares regression using normal equations"""
	
    b = tx.T.dot(y)
    # solve linear system using the QR decomposition
    return np.linalg.solve(A, b)
    
# ***************************************************
# REGRESSION
# ***************************************************
def ridge_regression(y, tx, lambda_):
    """Ridge regression using normal equations"""
    # We want to solve the linear system Ax = b
    # A is the sum of the Gram matrix and the identidy multiplied by lambda
    lambda_id = lambda_*np.identity(tx.shape[1])
    gram_mat = tx.T.dot(tx)
    A = gram_mat + lambda_id
    
    # b is the product between tx transposed and y
    b = tx.T.dot(y)
    
    # Solve with the QR decomposition
    return np.linalg.solve(a, b)

    
def logistic_regression(y, tx, initial_w, max_iters, gamma):
    """Logistic regression using gradient descent or SGD"""
    raise NotImplementedError


def reg_logistic_regression(y, tx, lambda_, initial_w, max_iters, gamma):
    """Regularized logistic regression using gradient descent or SGD"""    
    raise NotImplementedError