import numpy as np 
import Model.loadDataSet as loadDataSet 
import os
def load_quickdraw_dataset(folder_path, class_names, max_per_class=1000):
    X = []
    y = []
    for i, class_name in enumerate(class_names):
        filepath = os.path.join(folder_path, f"{class_name}.ndjson")
        images = loadDataSet.load_quickdraw_ndjson(filepath, max_items=max_per_class)
        X.append(images)
        y.append(np.full(len(images), i))
    X = np.concatenate(X, axis=0)
    y = np.concatenate(y, axis=0)
    return X, y

#helper function to one-hot encode the labels
def one_hot_encode(labels, num_classes):
    num_labels = labels.shape[0]
    one_hot = np.zeros((num_labels, num_classes))
    one_hot[np.arange(num_labels), labels] = 1
    return one_hot

#512,256,128 output: 25
def init_params(input_size, hidden_size,hidden2_size,hidden3_size, output_size):
    W1 = np.random.randn(input_size, hidden_size) * np.sqrt(2. / input_size) 
    b1 = np.zeros((1, hidden_size)) 

    W2 = np.random.randn(hidden_size, hidden2_size) *  np.sqrt(2. / hidden_size) 
    b2 = np.zeros((1, hidden2_size))

    W3 = np.random.randn(hidden2_size, hidden3_size) * np.sqrt(2. / hidden2_size)
    b3 = np.zeros((1, hidden3_size))

    W4 = np.random.randn(hidden3_size, output_size) *  np.sqrt(2. / hidden3_size)
    b4 = np.zeros((1, output_size))
    return W1, b1, W2, b2,W3, b3,W4,b4

#RELU activation function
def relu(x):
    return np.maximum(0,x)
def relu_derivative(x):
    return (x > 0).astype(float)

#Dropout function to prevent overfitting, removes a percentage of neurons every iteration
def dropout(a, dropout_rate):
    dropout_mask = (np.random.rand(*a.shape) > dropout_rate).astype(float)
    return (a * dropout_mask) / (1.0 - dropout_rate), dropout_mask

#Forward pass finding the output from previous layer and input for the next layer
#Setting dropout_rate to 0.2 by default, can be changed in the training script
def forward_pass(X, W1, b1, W2, b2, W3, b3, W4, b4, dropout_rate=0.2, training=True):
    z1 = np.dot(X, W1) + b1
    a1 = relu(z1)
    # Apply dropout if training and dropout rate is set
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

    return z1, a1, dropout_mask1, z2, a2, dropout_mask2, z3, a3, dropout_mask3, z4, output


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
        scailing_factor = clip_value / total_norm
        return [grad * scailing_factor for grad in gradients]
    return gradients

# Backward propagation to update the weights and biases
def backward_propagatation(X, y_true, z1, a1, dropout_mask1, z2, a2, dropout_mask2, z3, a3, dropout_mask3, z4, output, W2, W3, W4, dropout_rate=0.0):
    m = X.shape[0]

    delta4 = output - y_true
    dW4 = np.dot(a3.T, delta4) / m
    db4 = np.sum(delta4, axis=0, keepdims=True) / m

    delta3 = np.dot(delta4, W4.T) * relu_derivative(z3)
    delta3 *= dropout_mask3
    dW3 = np.dot(a2.T, delta3) / m
    db3 = np.sum(delta3, axis=0, keepdims=True) / m

    delta2 = np.dot(delta3, W3.T) * relu_derivative(z2)
    delta2 *= dropout_mask2
    dW2 = np.dot(a1.T, delta2) / m
    db2 = np.sum(delta2, axis=0, keepdims=True) / m

    delta1 = np.dot(delta2, W2.T) * relu_derivative(z1)
    delta1 *= dropout_mask1
    dW1 = np.dot(X.T, delta1) / m
    db1 = np.sum(delta1, axis=0, keepdims=True) / m

    return dW1, db1, dW2, db2, dW3, db3, dW4, db4


