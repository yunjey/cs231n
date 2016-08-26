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
  loss = 0.0
  dW = np.zeros_like(W)  # (3073, 10)

  #############################################################################
  # TODO: Compute the softmax loss and its gradient using explicit loops.     #
  # Store the loss in loss and the gradient in dW. If you are not careful     #
  # here, it is easy to run into numeric instability. Don't forget the        #
  # regularization!                                                           #
  #############################################################################
  
  y = (np.arange(W.shape[1]) == y[:, None]).astype(np.float32)  # (500, 10)
  h = np.dot(X, W)
  p = np.exp(h) / np.sum(np.exp(h), axis=1, keepdims=True) # softmax probabilities   (500, 10)
     

  s = np.multiply(p, y)  
  loss = np.sum(-np.log(s[s>0])) / X.shape[0]
  dy = p - y
  dW = np.dot(X.T, p) / X.shape[0]  # (3073, 500) x (500, 10) = (3073, 10)
  
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
  loss = 0.0
  dW = np.zeros_like(W)

  #############################################################################
  # TODO: Compute the softmax loss and its gradient using no explicit loops.  #
  # Store the loss in loss and the gradient in dW. If you are not careful     #
  # here, it is easy to run into numeric instability. Don't forget the        #
  # regularization!                                                           #
  #############################################################################
  y = (np.arange(W.shape[1]) == y[:, None]).astype(np.float32)  # (500, 10)
  h = np.dot(X, W)
  p = np.exp(h) / np.sum(np.exp(h), axis=1, keepdims=True) # softmax probabilities   (500, 10)

  s = np.multiply(p, y)  
  loss = np.sum(-np.log(s[s>0])) / X.shape[0]
  dy = p - y
  dW = np.dot(X.T, p) / X.shape[0] # (3073, 500) x (500, 10) = (3073, 10)
  #############################################################################
  #                          END OF YOUR CODE                                 #
  #############################################################################

  return loss, dW

