import numpy as np
from scipy.signal import correlate


# Utility functions
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

# Parameter initialization
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

    dummy = np.zeros((1, c, h, w), dtype=np.float32)
    A1 = relu(conv2d(dummy, params['W_conv1'], params['b_conv1']))
    P1 = max_pool(A1)
    A2 = relu(conv2d(P1, params['W_conv2'], params['b_conv2']))
    P2 = max_pool(A2)
    A3 = relu(conv2d(P2, params['W_conv3'], params['b_conv3']))
    P3 = max_pool(A3)
    flat_size = P3.reshape(1, -1).shape[1]

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



# Convolution helper
def prepVec(X, kernel_size, stride=1, padding=0):

    batch, channels, H, W = X.shape
    kh, kw = kernel_size, kernel_size

    X_pad = np.pad(X, ((0,0), (0,0), (padding,padding), (padding,padding)), mode="constant")
    H_p, W_p = X_pad.shape[2:]

    out_h = (H_p - kh)//stride + 1
    out_w = (W_p - kw)//stride + 1

    patches = np.zeros((batch, out_h, out_w, channels, kh, kw), dtype=X.dtype)
    for i in range(out_h):
        for j in range(out_w):
            patches[:, i, j, :, :, :] = X_pad[:, :, i*stride:i*stride+kh, j*stride:j*stride+kw]

    # Reshape to (batch, out_h*out_w, channels*kh*kw)
    patches = patches.reshape(batch, out_h*out_w, -1)
    return patches, out_h, out_w


def conv2d(X, W, b, stride=1, padding=0):
    batch, in_c, H, W_in = X.shape
    out_c, _, kh, kw = W.shape

    # Convert input to columns
    X_cols, out_h, out_w = prepVec(X, kh, stride, padding)  # (batch, out_h*out_w, in_c*kh*kw)
    W_col = W.reshape(out_c, -1)                           # (out_c, in_c*kh*kw)

    # Batch matmul
    out = X_cols @ W_col.T   # (batch, out_h*out_w, out_c)
    out = out + b.reshape(1, 1, -1)

    # Reshape back
    out = out.transpose(0, 2, 1).reshape(batch, out_c, out_h, out_w)
    return out
def max_pool(X, size=2):
    """
    2×2 max-pool that safely handles odd H/W by truncating the last row/col.
    Forward keeps only the largest full tiles; border leftovers are ignored.
    """
    b, c, h, w = X.shape
    h_eff = h - (h % size)
    w_eff = w - (w % size)
    if h_eff == 0 or w_eff == 0:
        raise ValueError(f"Input too small for pool size {size}: got {(h, w)}")

    Xc = X[:, :, :h_eff, :w_eff]                     # truncate odd border
    h_out, w_out = h_eff // size, w_eff // size
    Xr = Xc.reshape(b, c, h_out, size, w_out, size)
    out = Xr.max(axis=(3, 5))                        # pool (2,2)
    return out

def max_pool_backward(dOut, A, size=2):
    """
    Backward for the truncating 2×2 max-pool above.
    Only propagates gradients to the tiles used in forward (truncated area).
    """
    b, c, h, w = A.shape
    h_eff = h - (h % size)
    w_eff = w - (w % size)
    h_out, w_out = h_eff // size, w_eff // size

    # Work on the same truncated area used in forward
    Ac = A[:, :, :h_eff, :w_eff]
    dX = np.zeros_like(A, dtype=np.float32)
    dXc = dX[:, :, :h_eff, :w_eff]

    Ar = Ac.reshape(b, c, h_out, size, w_out, size)
    # Max along pooling axes, keep dims to build mask
    max_tiles = Ar.max(axis=(3, 5), keepdims=True)
    mask = (Ar == max_tiles)

    # Broadcast upstream grad into the pooling tiles
    dAr = np.zeros_like(Ar, dtype=np.float32)
    dAr += mask * dOut[:, :, :, None, :, None]

    # Fold back to image (still truncated area)
    dXc[:] = dAr.reshape(b, c, h_eff, w_eff)
    return dX


def forward_pass(X, params):
    caches = {}

    # --- Conv + Pool layers ---
    Z1 = conv2d(X, params['W_conv1'], params['b_conv1'])
    A1 = relu(Z1)
    P1 = max_pool(A1)
    caches['Z1'], caches['A1'], caches['P1'] = Z1, A1, P1

    Z2 = conv2d(P1, params['W_conv2'], params['b_conv2'])
    A2 = relu(Z2)
    P2 = max_pool(A2)
    caches['Z2'], caches['A2'], caches['P2'] = Z2, A2, P2

    Z3 = conv2d(P2, params['W_conv3'], params['b_conv3'])
    A3 = relu(Z3)
    P3 = max_pool(A3)
    caches['Z3'], caches['A3'], caches['P3'] = Z3, A3, P3

    # --- Flatten ---
    flat = P3.reshape(X.shape[0], -1)   
    caches['flat'] = flat

    # --- Fully connected layers ---
    Z_fc1 = flat @ params['W1'] + params['b1']
    A_fc1 = relu(Z_fc1)
    caches['Z_fc1'], caches['A_fc1'] = Z_fc1, A_fc1

    Z_fc2 = A_fc1 @ params['W2'] + params['b2']
    A_fc2 = relu(Z_fc2)
    caches['Z_fc2'], caches['A_fc2'] = Z_fc2, A_fc2

    Z_fc3 = A_fc2 @ params['W3'] + params['b3']
    A_fc3 = relu(Z_fc3)
    caches['Z_fc3'], caches['A_fc3'] = Z_fc3, A_fc3

    Z_out = A_fc3 @ params['W4'] + params['b4']
    probs = softmax(Z_out)
    caches['Z_out'], caches['probs'] = Z_out, probs

    return caches


# Backward pass helpers
def softmax_cross_entropy_derivative(A_out, Y):
    return (A_out - Y) / Y.shape[0]

def relu_back(dA, Z):
    return dA * (Z > 0)

def flatten_backward(dout, original_shape):
    return dout.reshape(original_shape)

def col2im(dcols, input_shape, kernel_size, stride=1, padding=0):

    b, in_c, H, W = input_shape
    kh = kw = kernel_size

    H_p, W_p = H + 2*padding, W + 2*padding
    out_h = (H_p - kh)//stride + 1
    out_w = (W_p - kw)//stride + 1

    dcols_resh = dcols.reshape(b, out_h, out_w, in_c, kh, kw)
    X_pad = np.zeros((b, in_c, H_p, W_p), dtype=dcols.dtype)

    for i in range(out_h):
        i0 = i*stride
        for j in range(out_w):
            j0 = j*stride
            X_pad[:, :, i0:i0+kh, j0:j0+kw] += dcols_resh[:, i, j, :, :, :]

    if padding > 0:
        return X_pad[:, :, padding:-padding, padding:-padding]
    return X_pad

def conv2d_Backward(X, W, b, stride=1, padding=0):
    bsz, in_c, h, w = X.shape
    out_c, _, k_h, k_w = W.shape

    # Pad input
    X_padded = np.pad(X, ((0,0), (0,0), (padding,padding), (padding,padding)), mode='constant')
    h_p, w_p = X_padded.shape[2], X_padded.shape[3]

    out_h = (h_p - k_h)//stride + 1
    out_w = (w_p - k_w)//stride + 1

    X_col = prepVec(X_padded, k_h, k_w, stride)
    W_col = W.reshape(out_c, -1)
    out = W_col @ X_col + b.reshape(-1,1)
    out = out.reshape(out_c, out_h, out_w, bsz).transpose(3,0,1,2)
    return out

def backward_pass(X, Y, params, caches):
    grads = {}
    m = X.shape[0]

    # --- Output layer ---
    dZ_out = caches['probs'] - Y
    grads['dW4'] = caches['A_fc3'].T @ dZ_out / m
    grads['db4'] = np.mean(dZ_out, axis=0)

    # --- Fully connected layers ---
    dA_fc3 = dZ_out @ params['W4'].T
    dZ_fc3 = dA_fc3 * relu_derivative(caches['Z_fc3'])
    grads['dW3'] = caches['A_fc2'].T @ dZ_fc3 / m
    grads['db3'] = np.mean(dZ_fc3, axis=0)

    dA_fc2 = dZ_fc3 @ params['W3'].T
    dZ_fc2 = dA_fc2 * relu_derivative(caches['Z_fc2'])
    grads['dW2'] = caches['A_fc1'].T @ dZ_fc2 / m
    grads['db2'] = np.mean(dZ_fc2, axis=0)

    dA_fc1 = dZ_fc2 @ params['W2'].T
    dZ_fc1 = dA_fc1 * relu_derivative(caches['Z_fc1'])
    grads['dW1'] = caches['flat'].T @ dZ_fc1 / m
    grads['db1'] = np.mean(dZ_fc1, axis=0)

    # --- Flatten to conv ---
    d_flat = dZ_fc1 @ params['W1'].T
    dP3 = d_flat.reshape(caches['P3'].shape)

    # --- Conv layer 3 ---
    dA3 = max_pool_backward(dP3, caches['A3'])
    dZ3 = dA3 * relu_derivative(caches['Z3'])
    grads['dW_conv3'], grads['db_conv3'] = conv_backward_single(caches['P2'], dZ3, params['W_conv3'])

    # --- Conv layer 2 ---
    dA2 = max_pool_backward(conv2d_Backward(dZ3, params['W_conv3'], stride=1, padding=0), caches['A2'])
    dZ2 = dA2 * relu_derivative(caches['Z2'])
    grads['dW_conv2'], grads['db_conv2'] = conv_backward_single(caches['P1'], dZ2, params['W_conv2'])

    # --- Conv layer 1 ---
    dA1 = max_pool_backward(conv2d_Backward(dZ2, params['W_conv2'], stride=1, padding=0), caches['A1'])
    dZ1 = dA1 * relu_derivative(caches['Z1'])
    grads['dW_conv1'], grads['db_conv1'] = conv_backward_single(X, dZ1, params['W_conv1'])

    return grads

def conv_backward_single(A_prev, dZ, W):
    """
    Compute gradients for a single conv layer (weights and biases).
    """
    batch, out_c, out_h, out_w = dZ.shape
    _, in_c, kh, kw = W.shape

    # Unfold input
    X_cols, _, _ = prepVec(A_prev, kh)
    dZ_reshaped = dZ.transpose(0, 2, 3, 1).reshape(batch, -1, out_c)

    dW = np.zeros_like(W)
    db = np.zeros(out_c)

    for i in range(out_c):
        db[i] = dZ[:, i, :, :].sum() / batch
        dW[i] = (X_cols.transpose(0, 2, 1) @ dZ_reshaped[:, :, i]).sum(axis=0).reshape(in_c, kh, kw) / batch

    return dW, db
