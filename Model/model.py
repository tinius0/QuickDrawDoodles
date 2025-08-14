import numpy as np

# ------------------------------
# Utility functions
# ------------------------------
def one_hot_encode(y, num_classes):
    y_encoded = np.zeros((len(y), num_classes), dtype=np.float32)
    y_encoded[np.arange(len(y)), y] = 1.0
    return y_encoded

def softmax(x):
    e_x = np.exp(x - np.max(x, axis=1, keepdims=True))
    return e_x / np.sum(e_x, axis=1, keepdims=True)

def relu(x):
    return np.maximum(0, x)

def relu_derivative(x):
    return (x > 0).astype(np.float32)

def cross_entropy(y_true, y_pred):
    eps = 1e-8
    y_pred = np.clip(y_pred, eps, 1-eps)
    return -np.mean(np.sum(y_true * np.log(y_pred), axis=1))

def clip_gradients(grads, max_norm):
    for k in grads.keys():
        grads[k] = np.clip(grads[k], -max_norm, max_norm)
    return grads

# ------------------------------
# Parameter initialization
# ------------------------------
def init_params(input_shape, conv_filters, hidden_sizes, output_size, kernel_size=3):
    """
    input_shape: (channels, H, W)
    conv_filters: list of 3 numbers [num_filters1, num_filters2, num_filters3]
    hidden_sizes: list of 3 numbers [hidden1, hidden2, hidden3]
    """
    params = {}
    c, h, w = input_shape
    # Convolution layers
    params['W_conv1'] = np.random.randn(conv_filters[0], c, kernel_size, kernel_size).astype(np.float32) * 0.01
    params['b_conv1'] = np.zeros(conv_filters[0], dtype=np.float32)
    
    params['W_conv2'] = np.random.randn(conv_filters[1], conv_filters[0], kernel_size, kernel_size).astype(np.float32) * 0.01
    params['b_conv2'] = np.zeros(conv_filters[1], dtype=np.float32)
    
    params['W_conv3'] = np.random.randn(conv_filters[2], conv_filters[1], kernel_size, kernel_size).astype(np.float32) * 0.01
    params['b_conv3'] = np.zeros(conv_filters[2], dtype=np.float32)
    
    # Compute flattened size after 3 conv layers (assuming stride=1, padding=0, 2x2 pooling)
    def conv_out(size, k=kernel_size, stride=1, pad=0):
        return (size - k + 2*pad)//stride + 1
    def pool_out(size, pool=2):
        return size // pool

    h1 = pool_out(conv_out(h))
    w1 = pool_out(conv_out(w))
    h2 = pool_out(conv_out(h1))
    w2 = pool_out(conv_out(w1))
    h3 = pool_out(conv_out(h2))
    w3 = pool_out(conv_out(w2))
    flat_size = conv_filters[2]*h3*w3

    # Fully connected layers
    params['W1'] = np.random.randn(flat_size, hidden_sizes[0]).astype(np.float32) * np.sqrt(2/flat_size)
    params['b1'] = np.zeros(hidden_sizes[0], dtype=np.float32)
    params['W2'] = np.random.randn(hidden_sizes[0], hidden_sizes[1]).astype(np.float32) * np.sqrt(2/hidden_sizes[0])
    params['b2'] = np.zeros(hidden_sizes[1], dtype=np.float32)
    params['W3'] = np.random.randn(hidden_sizes[1], hidden_sizes[2]).astype(np.float32) * np.sqrt(2/hidden_sizes[1])
    params['b3'] = np.zeros(hidden_sizes[2], dtype=np.float32)
    params['W4'] = np.random.randn(hidden_sizes[2], output_size).astype(np.float32) * np.sqrt(2/hidden_sizes[2])
    params['b4'] = np.zeros(output_size, dtype=np.float32)

    return params

# ------------------------------
# Convolution helper
# ------------------------------
def conv2d(X, W, b, stride=1, padding=0):
    batch, in_c, h, w = X.shape
    out_c, _, kh, kw = W.shape
    h_out = (h - kh + 2*padding)//stride + 1
    w_out = (w - kw + 2*padding)//stride + 1
    X_pad = np.pad(X, ((0,0),(0,0),(padding,padding),(padding,padding)), mode='constant')
    out = np.zeros((batch, out_c, h_out, w_out), dtype=np.float32)
    for n in range(batch):
        for c in range(out_c):
            for i in range(h_out):
                for j in range(w_out):
                    patch = X_pad[n,:, i*stride:i*stride+kh, j*stride:j*stride+kw]
                    out[n,c,i,j] = np.sum(patch*W[c]) + b[c]
    return out

def max_pool(X, size=2):
    batch, c, h, w = X.shape
    h_out = h//size
    w_out = w//size
    out = np.zeros((batch,c,h_out,w_out), dtype=np.float32)
    for i in range(h_out):
        for j in range(w_out):
            patch = X[:,:, i*size:i*size+size, j*size:j*size+size]
            out[:,:,i,j] = np.max(patch, axis=(2,3))
    return out

# ------------------------------
# Forward pass
# ------------------------------
def forward_pass(X, params):
    caches = {}
    caches['X'] = X
    # Conv1
    Z1 = conv2d(X, params['W_conv1'], params['b_conv1'])
    A1 = relu(Z1)
    P1 = max_pool(A1)
    caches.update({'Z1': Z1, 'A1': A1, 'P1': P1})

    # Conv2
    Z2 = conv2d(P1, params['W_conv2'], params['b_conv2'])
    A2 = relu(Z2)
    P2 = max_pool(A2)
    caches.update({'Z2': Z2, 'A2': A2, 'P2': P2})

    # Conv3
    Z3 = conv2d(P2, params['W_conv3'], params['b_conv3'])
    A3 = relu(Z3)
    P3 = max_pool(A3)
    caches.update({'Z3': Z3, 'A3': A3, 'P3': P3})

    # Flatten
    batch = X.shape[0]
    flat = P3.reshape(batch, -1)
    caches['flat'] = flat

    # Fully connected layers
    Z_fc1 = flat @ params['W1'] + params['b1']
    A_fc1 = relu(Z_fc1)
    Z_fc2 = A_fc1 @ params['W2'] + params['b2']
    A_fc2 = relu(Z_fc2)
    Z_fc3 = A_fc2 @ params['W3'] + params['b3']
    A_fc3 = relu(Z_fc3)
    Z_out = A_fc3 @ params['W4'] + params['b4']
    A_out = softmax(Z_out)
    caches.update({
        'Z_fc1': Z_fc1, 'A_fc1': A_fc1,
        'Z_fc2': Z_fc2, 'A_fc2': A_fc2,
        'Z_fc3': Z_fc3, 'A_fc3': A_fc3,
        'Z_out': Z_out, 'A_out': A_out
    })
    return caches
# ------------------------------
# Backward pass helpers
# ------------------------------
def softmax_cross_entropy_derivative(A_out, Y):
    return (A_out - Y) / Y.shape[0]

def relu_back(dA, Z):
    return dA * (Z > 0)

def flatten_backward(dout, original_shape):
    return dout.reshape(original_shape)

def max_pool_backward(dA, A, size=2):
    batch, c, h_out, w_out = dA.shape
    dX = np.zeros_like(A, dtype=np.float32)
    for i in range(h_out):
        for j in range(w_out):
            patch = A[:, :, i*size:i*size+size, j*size:j*size+size]
            mask = (patch == np.max(patch, axis=(2,3), keepdims=True))
            dX[:, :, i*size:i*size+size, j*size:j*size+size] += mask * dA[:, :, i, j][:, :, None, None]
    return dX

def conv2d_backward(dout, X, W, stride=1, padding=0):
    batch, in_c, h, w = X.shape
    out_c, _, kh, kw = W.shape
    h_out, w_out = dout.shape[2], dout.shape[3]
    X_pad = np.pad(X, ((0,0),(0,0),(padding,padding),(padding,padding)), mode='constant')
    dX_pad = np.zeros_like(X_pad)
    dW = np.zeros_like(W)
    db = np.zeros_like(W[:,0,0,0])
    for n in range(batch):
        for c in range(out_c):
            for i in range(h_out):
                for j in range(w_out):
                    patch = X_pad[n, :, i*stride:i*stride+kh, j*stride:j*stride+kw]
                    dW[c] += dout[n, c, i, j] * patch
                    dX_pad[n, :, i*stride:i*stride+kh, j*stride:j*stride+kw] += dout[n, c, i, j] * W[c]
            db[c] += np.sum(dout[n, c])
    if padding > 0:
        dX = dX_pad[:, :, padding:-padding, padding:-padding]
    else:
        dX = dX_pad
    return dX, dW, db
def backward_pass(caches, params, Y):
    grads = {}

    # Output layer
    dZ_out = softmax_cross_entropy_derivative(caches['A_out'], Y)
    grads['W4'] = caches['A_fc3'].T @ dZ_out
    grads['b4'] = np.sum(dZ_out, axis=0)
    dA3 = dZ_out @ params['W4'].T

    # FC3
    dZ3 = relu_back(dA3, caches['Z_fc3'])
    grads['W3'] = caches['A_fc2'].T @ dZ3
    grads['b3'] = np.sum(dZ3, axis=0)
    dA2 = dZ3 @ params['W3'].T

    # FC2
    dZ2 = relu_back(dA2, caches['Z_fc2'])
    grads['W2'] = caches['A_fc1'].T @ dZ2
    grads['b2'] = np.sum(dZ2, axis=0)
    dA1 = dZ2 @ params['W2'].T

    # FC1
    dZ1 = relu_back(dA1, caches['Z_fc1'])
    grads['W1'] = caches['flat'].T @ dZ1
    grads['b1'] = np.sum(dZ1, axis=0)
    dflat = dZ1 @ params['W1'].T

    # Reshape back to conv output
    dP3 = flatten_backward(dflat, caches['P3'].shape)
    dA3_conv = max_pool_backward(dP3, caches['A3'])
    dZ3_conv = relu_back(dA3_conv, caches['Z3'])
    dP2, grads['W_conv3'], grads['b_conv3'] = conv2d_backward(dZ3_conv, caches['P2'], params['W_conv3'])

    dA2_conv = max_pool_backward(dP2, caches['A2'])
    dZ2_conv = relu_back(dA2_conv, caches['Z2'])
    dP1, grads['W_conv2'], grads['b_conv2'] = conv2d_backward(dZ2_conv, caches['P1'], params['W_conv2'])

    dA1_conv = max_pool_backward(dP1, caches['A1'])
    dZ1_conv = relu_back(dA1_conv, caches['Z1'])
    _, grads['W_conv1'], grads['b_conv1'] = conv2d_backward(dZ1_conv, caches['X'], params['W_conv1'])

    # Clip gradients
    grads = clip_gradients(grads, max_norm=1.0)
    return grads
