import numpy as np


# forward: fully-connected layer
def affine_forward(X, W, b):
    
    out = np.dot(X, W) + b
    cache = (X, w, b)
    return out, cache


# backward: fully-connected layer
def affine_backward(dout, cache):
    
    X, W, b = cache
    dx = np.dot(dout, W.T)
    dw = np.dot(X.T, dout) 
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

# forward: convolutional layer
def conv_forward(X, w, b, conv_param):
    
    N, C, H, W = x.shape                 # (batch size, channel size, height, width)
    K, _, F, _ = w.shape                 # (number of filters, channel size, filter height, filter width)
    S = conv_param['stride']             # stride size
    P = conv_param['pad']                # pad size
    H_out = (H - F + 2 * P) / S + 1      # output height
    W_out = (W - F + 2 * P) / S + 1      # output width
    out = np.zeros((N, K, H_out, W_out))
    
    x_pad = np.pad(x, ((0, 0), (0, 0), (P, P), (P, P)), mode='constant') # pad spacially
  
    for i in range(N):
        image = x_pad[i, :, : ,:]           # choose one image 
        for j in range(K):                  # choose one filter
            for k in range(H_out):  
                for l in range(W_out):
                    image_patch = image[:, (k*S):(k*S + F), (l*S):(l*S + F)] 
                    out[i, j, k, l] = np.sum(np.multiply(image_patch, w[j, :, :, :])) + b[j] 
                    
    cache = (x, w, b, conv_param)
    return out, cache


# backward: convolutional layer
def conv_backward(dout, cache):
    
    x, w, b, conv_param = cache
    _, C, F, _ = w.shape                    # (number of filters, channel size, filter height, filter width)
    N, K, H_out, W_out = dout.shape         # (batch size, number of filters, output height, ouput width)
    S = conv_param['stride']                # stride size
    P = conv_param['pad']                   # pad size
    x_pad = np.pad(x, [(0, 0), (0, 0), (P, P), (P, P)], mode="constant")  
    dw, db, dx = np.zeros_like(w), np.zeros_like(b), np.zeros_like(x_pad)

    for i in range(N):
        image = x_pad[i, :, :, :]           # choose one image 
        for j in range(K):                  # choose one filter
            for k in range(H_out):
                for l in range(W_out):
                    image_patch = image[:, (k*S):(k*S + F), (l*S):(l*S + F)]
                    dw[j, :, :, :] += image_patch * dout[i, j, k, l]
                    db[j] += dout[i, j, k, l]
                    dx[i, :, (k*S):(k*S + F), (l*S):(l*S + F)] += w[j, :, :, :] * dout[i, j, k, l]
        
    dx = dx[:, :, P:-P, P:-P] # delete padding

    return dx, dw, db
    
    
    
    
    