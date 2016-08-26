import numpy as np

# forward: fully-connected layer
def affine_forward(X, W, b):
    
    out = np.dot(X, W) + b
    cache = (X, w, b)
    
    return out, cache


# backward: fully-connected layer
def affine_bakward(dout, cache):
    
    X, W, b = cache
    
    dx = np.dot(dout, W.T)
    dw = np.dot(X.T, dout) / X.shape[0]
    db = np.sum(dout, axis=0)
    
    
    return dx, dw, db


# forward: relu
def relu_forward(X):
    
    out = np.maximum(X, 0)
    cache = X
    
    return out, cache


# backward: relu
def relu_backward(dout, cache):
    
    dx = dout * (cache > 0)
    
    return dx
    

# forward: softmax 
def softmax_forward(X):
    
    p = np.exp(x) / np.sum(np.exp(x),axis=1,keepdims=True)
    
    return p


# backward: softmax
def softmax_backward(X, y):
    
    N = X.shape[0]
    dx = np.copy(X)
    dx[np.arange(N), y] -= 1 
    
    return dx






    
    
    
    
    
    