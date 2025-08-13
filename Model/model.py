import numpy as np 
import loadDataSet 
import os
from sklearn.model_selection import train_test_split

# Helper functions for the CNN layers
def conv_forward(X, W, b, stride=1, pad=2):
    n, h, w, c = X.shape
    num_filters, filter_h, filter_w, c = W.shape
    
    out_h = int(1 + (h + 2 * pad - filter_h) / stride)
    out_w = int(1 + (w + 2 * pad - filter_w) / stride)

    X_pad = np.pad(X, ((0, 0), (pad, pad), (pad, pad), (0, 0)), 'constant')
    out = np.zeros((n, out_h, out_w, num_filters))

    for i in range(out_h):
        for j in range(out_w):
            X_slice = X_pad[:, i*stride:i*stride+filter_h, j*stride:j*stride+filter_w, :]
            for k in range(num_filters):
                out[:, i, j, k] = np.sum(X_slice * W[k, :, :, :], axis=(1, 2, 3)) + b[k]
    return out

# Max pooling layer to reduce dimensions and extract features from the previous layer
def max_pool_forward(X, pool_h=2, pool_w=2, stride=2):
    n, h, w, c = X.shape
    out_h = int(1 + (h - pool_h) / stride)
    out_w = int(1 + (w - pool_w) / stride)
    
    out = np.zeros((n, out_h, out_w, c))

    for i in range(out_h):
        for j in range(out_w):
            X_slice = X[:, i*stride:i*stride+pool_h, j*stride:j*stride+pool_w, :]
            out[:, i, j, :] = np.max(X_slice, axis=(1, 2))
    return out


def load_quickdraw_dataset(folder_path, class_names, max_per_class=30000):
    X = []
    y = []
    for i, class_name in enumerate(class_names):
        filepath = os.path.join(folder_path, f"{class_name}.ndjson")
        images = loadDataSet.load_quickdraw_ndjson(filepath, max_items=max_per_class)
        X.append(images)
        y.append(np.full(len(images), i))
    X = np.concatenate(X, axis=0)
    y = np.concatenate(y, axis=0)
    X = X.astype(np.float32)  
    return X, y

#helper function to one-hot encode the labels
def one_hot_encode(labels, num_classes):
    num_labels = labels.shape[0]
    one_hot = np.zeros((num_labels, num_classes))
    one_hot[np.arange(num_labels), labels] = 1
    return one_hot

#Heuristic initialization for weights, using He initialization
def he_init(shape):
    fan_in = shape[0]
    return np.random.randn(*shape) * np.sqrt(2. / fan_in)


#512,256,128 output: 3
def init_params(input_size, hidden_size,hidden2_size,hidden3_size, output_size,num_filters):
    # CNN Layer parameters
    conv_filter_h, conv_filter_w = 5, 5
    W_conv = np.random.randn(num_filters, conv_filter_h, conv_filter_w, 1) * np.sqrt(2. / (5 * 5 * 1))
    b_conv = np.zeros(num_filters)
    
    # Calculate the size of the output from the CNN layers
    conv_out_h = int(1 + (28 + 2 * 2 - 5) / 1)
    conv_out_w = int(1 + (28 + 2 * 2 - 5) / 1)
    
    # After max pooling 
    pool_out_h = int(1 + (conv_out_h - 2) / 2)
    pool_out_w = int(1 + (conv_out_w - 2) / 2)
    
    # The flattened size is the input to the first dense layer
    flattened_size = pool_out_h * pool_out_w * num_filters

    #Initialize weights and biases for the dense layers
    W1 = he_init((flattened_size, hidden_size))  
    b1 = np.zeros((1, hidden_size))

    W2 = he_init((hidden_size, hidden2_size))
    b2 = np.zeros((1, hidden2_size))

    W3 = he_init((hidden2_size, hidden3_size))
    b3 = np.zeros((1, hidden3_size))

    #Xavier initialization for output layer
    W4 = np.random.randn(hidden3_size, output_size) * np.sqrt(1. / hidden3_size)
    b4 = np.zeros((1, output_size))

    return W_conv, b_conv, W1, b1, W2, b2, W3, b3, W4, b4

#RELU activation function
def relu(x):
    return np.maximum(0,x)

def relu_derivative(x):
    return (x > 0).astype(float)

#Dropout function to prevent overfitting, removes a percentage of neurons every iteration
def dropout(a, dropout_rate):
    if dropout_rate == 0.0:
        return a, np.ones_like(a, dtype=np.float32)
    mask = (np.random.rand(*a.shape) > dropout_rate).astype(np.float32)
    mask = mask / (1.0 - dropout_rate)   # scale the mask
    return a * mask, mask

#Forward pass finding the output from previous layer and input for the next layer
#Setting dropout_rate to 0.2 by default, can be changed in the training script
def forward_pass(X, W_conv, b_conv, W1, b1, W2, b2, W3, b3, W4, b4, dropout_rate=0.3, training=True):
    X_reshaped = X.reshape(-1, 28, 28, 1)
    Z_conv = conv_forward(X_reshaped, W_conv, b_conv)
    A_conv = relu(Z_conv)
    A_pool = max_pool_forward(A_conv)
    flattened = A_pool.reshape(A_pool.shape[0], -1)

    z1 = np.dot(flattened, W1) + b1
    a1 = relu(z1)
    if training and dropout_rate > 0.0:
        a1, dropout_mask1 = dropout(a1, dropout_rate)
    else:
        dropout_mask1 = np.ones_like(a1)

    z2 = np.dot(a1, W2) + b2
    a2 = relu(z2)
    if training and dropout_rate > 0.0:
        a2, dropout_mask2 = dropout(a2, dropout_rate)
    else:
        dropout_mask2 = np.ones_like(a2)

    z3 = np.dot(a2, W3) + b3
    a3 = relu(z3)
    if training and dropout_rate > 0.0:
        a3, dropout_mask3 = dropout(a3, dropout_rate)
    else:
        dropout_mask3 = np.ones_like(a3)

    z4 = np.dot(a3, W4) + b4
    output = softmax(z4)

    return X_reshaped, Z_conv, A_conv, A_pool, flattened, z1, a1, dropout_mask1, z2, a2, dropout_mask2, z3, a3, dropout_mask3, z4, output


#Output layer activation function
def softmax(x):
    exp_x = np.exp(x - np.max(x, axis=1, keepdims=True))
    return exp_x / np.sum(exp_x, axis=1, keepdims=True)

#Cross-entropy to fine tune the model, calculating the loss
def cross_entropy(y_true, y_pred):
    epsilon = 1e-12
    y_pred = np.clip(y_pred, epsilon, 1. - epsilon)
    loss = -np.sum(y_true * np.log(y_pred)) / y_true.shape[0]
    return loss

#QuickDraw is a noisy dataset, so we implement gradient clipping to prevent huge gradients
def clip_gradients(gradients, clip_value):
    total_norm = 0
    for gradient in gradients:
        total_norm += np.sum(np.square(gradient))
    total_norm = np.sqrt(total_norm)

    if total_norm > clip_value:
        scaling_factor = clip_value / total_norm
        return [grad * scaling_factor for grad in gradients]
    return gradients

# Backward propagation to update the weights and biases
def backward_propagation(X_reshaped, Z_conv, A_conv, A_pool, flattened, Z1, A1, dropout_mask1, Z2, A2, dropout_mask2, Z3, A3, dropout_mask3, Z4, output, y_true, W_conv, b_conv, W1, W2, W3, W4, b1, b2, b3, b4, dropout_rate):

    m = y_true.shape[0]
    dZ4 = (output - y_true) / m

    dW4 = np.dot(A3.T, dZ4)
    db4 = np.sum(dZ4, axis=0, keepdims=True)
    dA3 = np.dot(dZ4, W4.T)

    if dropout_rate > 0.0:
        dA3 *= dropout_mask3
    dZ3 = dA3 * relu_derivative(Z3)

    dW3 = np.dot(A2.T, dZ3)
    db3 = np.sum(dZ3, axis=0, keepdims=True)
    dA2 = np.dot(dZ3, W3.T)

    if dropout_rate > 0.0:
        dA2 *= dropout_mask2
    dZ2 = dA2 * relu_derivative(Z2)

    dW2 = np.dot(A1.T, dZ2)
    db2 = np.sum(dZ2, axis=0, keepdims=True)
    dA1 = np.dot(dZ2, W2.T)

    if dropout_rate > 0.0:
        dA1 *= dropout_mask1
    dZ1 = dA1 * relu_derivative(Z1)

    dW1 = np.dot(flattened.T, dZ1)
    db1 = np.sum(dZ1, axis=0, keepdims=True)

    # Gradient for flattened input from the pool layer
    dflattened = np.dot(dZ1, W1.T)

    # Reshape gradient to shape of A_pool
    dA_pool = dflattened.reshape(A_pool.shape)

    # Backprop max pooling
    dA_conv = np.zeros_like(A_conv)
    pool_h, pool_w, stride = 2, 2, 2
    n, out_h, out_w, c = A_pool.shape
    for i in range(out_h):
        for j in range(out_w):
            slice_ = A_conv[:, i*stride:i*stride+pool_h, j*stride:j*stride+pool_w, :]
            max_mask = (slice_ == np.max(slice_, axis=(1,2), keepdims=True))
            dA_conv[:, i*stride:i*stride+pool_h, j*stride:j*stride+pool_w, :] += dA_pool[:, i:i+1, j:j+1, :] * max_mask

    # Backprop conv relu
    dZ_conv = dA_conv * relu_derivative(Z_conv)

    # Gradients for conv weights and biases
    n, h, w, c = X_reshaped.shape
    num_filters, filter_h, filter_w, c = W_conv.shape

    dW_conv = np.zeros_like(W_conv)
    db_conv = np.zeros_like(b_conv)

    X_pad = np.pad(X_reshaped, ((0,0),(2,2),(2,2),(0,0)), 'constant')
    dZ_pad = np.pad(dZ_conv, ((0,0),(2,2),(2,2),(0,0)), 'constant')

    for k in range(num_filters):
        for i in range(dZ_conv.shape[1]):
            for j in range(dZ_conv.shape[2]):
                X_slice = X_pad[:, i:i+filter_h, j:j+filter_w, :]
                dZ_slice = dZ_conv[:, i, j, k][:, None, None, None]
                dW_conv[k, :, :, :] += np.sum(X_slice * dZ_slice, axis=0)

        db_conv[k] = np.sum(dZ_conv[:, :, :, k])

    return dW_conv, db_conv, dW1, db1, dW2, db2, dW3, db3, dW4, db4

def update_params(params, grads, learning_rate):
    return [param - learning_rate * grad for param, grad in zip(params, grads)]