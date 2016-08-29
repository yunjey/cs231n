import numpy as np
from random import shuffle

def softmax_loss_naive(W, X, y, reg):
  """
  Softmax loss function, naive implementation (with loops)

  Inputs have dimension D, there are C classes, and we operate on minibatches
  of N examples.

  Inputs:
  - W: A numpy array of shape (D, C) containing weights.
  - X: A numpy array of shape (N, D) containing a minibatch of data.
  - y: A numpy array of shape (N,) containing training labels; y[i] = c means
    that X[i] has label c, where 0 <= c < C.
  - reg: (float) regularization strength

  Returns a tuple of:
  - loss as single float
  - gradient with respect to weights W; an array of same shape as W
  """
  # Initialize the loss and gradient to zero.
  N, D = X.shape
  C = W.shape[1]

  loss = 0.0
  out = np.zeros((N,C))
  dW = np.zeros_like(W)  # (3073, 10)
  
  #############################################################################
  # TODO: Compute the softmax loss and its gradient using explicit loops.     #
  # Store the loss in loss and the gradient in dW. If you are not careful     #
  # here, it is easy to run into numeric instability. Don't forget the        #
  # regularization!                                                           #
  #############################################################################
  
  # forward
  for i in range(N):
    for j in range(C):
      for k in range(D):
        out[i, j] += X[i, k] * W[k, j]
    out[i, :] = np.exp(out[i, :])
    out[i, :] /= np.sum(out[i, :])  #  (N, C)
  
  # compute loss
  loss -= np.sum(np.log(out[np.arange(N), y])) 
  loss /= N
  loss += 0.5 * reg * np.sum(W**2)
  
  # backward
  out[np.arange(N), y] -= 1   # (N, C)
 
  for i in range(N):
    for j in range(D):
      for k in range(C):
        dW[j, k] += X[i, j] * out[i, k] 

  # add reg term
  dW /= N
  dW += reg * W

  #############################################################################
  #                          END OF YOUR CODE                                 #
  #############################################################################

  return loss, dW


def softmax_loss_vectorized(W, X, y, reg):
  """
  Softmax loss function, vectorized version.

  Inputs and outputs are the same as softmax_loss_naive.
  """
  # Initialize the loss and gradient to zero.
  N = X.shape[0]
  loss = 0.0
  dW = np.zeros_like(W)

  #############################################################################
  # TODO: Compute the softmax loss and its gradient using no explicit loops.  #
  # Store the loss in loss and the gradient in dW. If you are not careful     #
  # here, it is easy to run into numeric instability. Don't forget the        #
  # regularization!                                                           #
  #############################################################################
  
  # forward
  score = np.dot(X, W)   # (N, C)
  out = np.exp(score)
  out /= np.sum(out, axis=1, keepdims=True)   # (N, C)
  loss -= np.sum(np.log(out[np.arange(N), y]))
  loss /= N
  loss += 0.5 * reg * np.sum(W**2)

  # backward
  dout = np.copy(out)   # (N, C)
  dout[np.arange(N), y] -= 1
  dW = np.dot(X.T, dout)  # (D, C)
  dW /= N
  dW += reg * W
    
  #############################################################################
  #                          END OF YOUR CODE                                 #
  #############################################################################

  return loss, dW

