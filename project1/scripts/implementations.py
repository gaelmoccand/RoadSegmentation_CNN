# -*- coding: utf-8 -*-
"""Useful methods for project 1"""
import numpy as np

# ***************************************************
# COSTS
# ***************************************************
def calculate_mse(e):
    """Calculate the mse from the input vector"""
    N = len(e)
    mse = e.T.dot(e)/(2*N)
    return mse
    
def compute_loss(y, tx, w):
    """Calculate the loss using mse"""
    error = y - tx.dot(w)
    return calculate_mse(error)

# ***************************************************
# GRADIENT DESCENT
# ***************************************************
def compute_gradient(y, tx, w):
    """Compute the gradient."""
    error = y - tx.dot(w)
    N = len(error)
    grad = -tx.T.dot(error)/N
    return grad, error

# ***************************************************
# LEAST SQUARES
# ***************************************************    
def least_squares_GD(y, tx, initial_w, max_iters, gamma):
    """Linear regression using gradient descent"""
    w = initial_w
    for n_iter in range(max_iters):
        grad,_ = compute_gradient(y,tx,w)
        w = w - gamma*grad
        #print("GD ({bi}/{ti}): loss={l}, w0={w0}, w1={w1}".format(
              #bi=n_iter, ti=max_iters - 1, l=loss, w0=w[0], w1=w[1]))    
    loss = compute_loss(y,tx,w)
    return w,loss


def least_squares_SGD(y, tx, initial_w, max_iters, gamma):
    """Linear regression using stochastic gradient descent"""
    w = initial_w
    for n_iter in range(max_iters):
        for minibatch_y, minibatch_tx in batch_iter(y, tx, 1): # set batch size to 128
            grad,_=compute_gradient(minibatch_y,minibatch_tx,w) # compute the stochastic gradient using the minibatches
            w = w - gamma*grad # update the w
            #print("SGD ({bi}/{ti}): loss={l}, w0={w0}, w1={w1}".format(
             # bi=n_iter, ti=max_iters - 1, l=loss, w0=w[0], w1=w[1]))   
    loss = compute_loss(y,tx,w)# compute the loss using the entire sets
    return w,loss
    

def least_squares(y, tx):
    """Least squares regression using normal equations"""
    # We want to solve the linear system Aw = b...
    # ...with A being the Gram Matrix...
    A = tx.T.dot(tx)
    # ... and b being the transpose of tx times y
    b = tx.T.dot(y)
    # solve linear system using the QR decomposition
    w=np.linalg.solve(A, b)
    loss = compute_loss(y,tx,w)# compute the loss using the entire sets
    return w,loss
    
# ***************************************************
# REGRESSION RIDGE + Logistics
# ***************************************************
def ridge_regression(y, tx, lambda_):
    """Ridge regression using normal equations"""
    # We want to solve the linear system Ax = b
    # A is the sum of the Gram matrix and the identidy multiplied by lambda
    lambda_id = tx.shape[0]*lambda_*np.identity(tx.shape[1])
    gram_mat = tx.T.dot(tx)
    A = gram_mat + lambda_id
    # b is the product between tx transposed and y
    b = tx.T.dot(y)
    # Solve with the QR decomposition
    w=np.linalg.solve(A, b)
    loss = compute_loss(y,tx,w)# compute the loss using the entire sets
    return w,loss
    
def logistic_regression(y, tx, initial_w, max_iters, gamma): #FIXME diverge
    """Logistic regression using gradient descent"""
    losses = []
    w = initial_w
    for n_iter in range(max_iters):
        Xw = tx.dot(w);
        sigma = np.exp(Xw)/(1 + np.exp(Xw));
        # Compute the gradient of the loss w.r.t w
        grad = tx.T.dot(sigma - y)
        # Update w by gradient
        w = w - gamma*grad # update the w
        #print("Gradient Descent({bi}/{ti}): loss={l}, w0={w0}, w1={w1}".format(
              #bi=n_iter, ti=max_iters - 1, l=loss, w0=w[0], w1=w[1]
    loss = compute_loss(y,tx,w)# compute the loss using the entire sets
    return w,loss  

def reg_logistic_regression(y, tx, lambda_, initial_w, max_iters, gamma): #FIXME diverge
    """Regularized logistic regression using gradient descent"""    
    # Define parameters to store w and loss
    ws = [initial_w]
    losses = []
    w = initial_w
    for n_iter in range(max_iters):
        Xw = tx.dot(w)/1000;
        sigma = np.exp(Xw)/(1 + np.exp(Xw))
        print(Xw)
        # Compute the gradient of the loss w.r.t w
        grad = tx.T.dot(sigma - y) + lambda_*w
        # Update w by gradient
        w = w - gamma*grad # update the w
        #print("Gradient Descent({bi}/{ti}): loss={l}, w0={w0}, w1={w1}".format(
              #bi=n_iter, ti=max_iters - 1, l=loss, w0=w[0], w1=w[1]
    loss = compute_loss(y,tx,w)# compute the loss using the entire sets
    return w,loss


    
# ***************************************************
# ffrom Helpers
# ***************************************************

def batch_iter(y, tx, batch_size, num_batches=1, shuffle=True):
    """
    Generate a minibatch iterator for a dataset.
    Takes as input two iterables (here the output desired values 'y' and the input data 'tx')
    Outputs an iterator which gives mini-batches of `batch_size` matching elements from `y` and `tx`.
    Data can be randomly shuffled to avoid ordering in the original data messing with the randomness of the minibatches.
    Example of use :
    for minibatch_y, minibatch_tx in batch_iter(y, tx, 32):
        <DO-SOMETHING>
    """
    data_size = len(y)

    if shuffle:
        shuffle_indices = np.random.permutation(np.arange(data_size))
        shuffled_y = y[shuffle_indices]
        shuffled_tx = tx[shuffle_indices]
    else:
        shuffled_y = y
        shuffled_tx = tx
    for batch_num in range(num_batches):
        start_index = batch_num * batch_size
        end_index = min((batch_num + 1) * batch_size, data_size)
        if start_index != end_index:
            yield shuffled_y[start_index:end_index], shuffled_tx[start_index:end_index]
