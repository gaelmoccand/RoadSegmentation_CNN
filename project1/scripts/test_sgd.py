from implemenations import *
from helpers import *
import datetime


def least_squares_SGD(y, tx, initial_w, max_iters, gamma):
    """Linear regression using stochastic gradient descent"""

    # Define parameters to store w and loss
    ws = [initial_w]
    losses = []
    w = initial_w
    for n_iter in range(max_iters):
        for minibatch_y, minibatch_tx in batch_iter(y, tx, 128): # set batch size to 128
            grad,_=compute_gradient(minibatch_y,minibatch_tx,w) # compute the stochastic gradient using the minibatches
            w = w - (gamma)*grad # update the w
            loss = compute_loss(y,tx,w)# compute the loss using the entire sets
            ws.append(w)#save w
            losses.append(loss) #save the loss
        print("Stochastic Gradient Descent({bi}/{ti}): loss={l}, w0={w0}, w1={w1}".format(
              bi=n_iter, ti=max_iters - 1, l=loss, w0=w[0], w1=w[1]))    
    return losses, ws


