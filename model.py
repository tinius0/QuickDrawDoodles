import numpy as np
from keras.datasets import mnist

mnistDataSet = mnist.load_data()

#helper function to one-hot encode the labels
def one_hot_encode(labels, num_classes):
    num_labels = labels.shape[0]
    one_hot = np.zeros((num_labels, num_classes))
    one_hot[np.arange(num_labels), labels] = 1
    return one_hot
# Load the MNIST dataset from mnistDataSet variable
def load_mnist_images():
    (x_train, _), (x_test, _) = mnistDataSet
    x_train = x_train.reshape(-1, 28*28) / 255.0
    x_test = x_test.reshape(-1, 28*28) / 255.0
    return x_train, x_test

def load_mnist_labels():
    (_, y_train), (_, y_test) = mnistDataSet
    #Remember to one-hot encode the labels
    return y_train, y_test

def init_params(input_size, hidden_size,hidden2_size, output_size):
    W1 = np.random.randn(input_size, hidden_size) * 0.01 
    b1 = np.zeros((1, hidden_size)) 

    W2 = np.random.randn(hidden_size, hidden2_size) * 0.01  
    b2 = np.zeros((1, hidden2_size))

    W3 = np.random.randn(hidden2_size, output_size) *0.01
    b3 = np.zeros((1, output_size))

    return W1, b1, W2, b2,W3, b3

#RELU activation function
def relu(x):
    return np.maximum(0,x)
def relu_derivative(x):
    return (x > 0).astype(float)


#Forward pass finding the output from previous layer and input for the next layer
def forward_pass(X,W1,b1,W2,b2,W3,b3):
    z1 = np.dot(X,W1)+ b1 
    a1 = relu(z1) #Compress the matrix multiplication result to a value between 0 and 1

    z2 = np.dot(a1,W2) + b2
    a2 = relu(z2)

    z3 = np.dot(a2,W3) +b3
    output = softmax(z3)
    return z1,a1,z2,a2,z3, output

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

# Backward propagation to update the weights and biases
def backward_propagatation(X, y_true, z1, a1, z2, a2, z3, output, W2, W3):
    m = X.shape[0]

    delta3 = output - y_true
    dW3 = np.dot(a2.T, delta3) / m
    db3 = np.sum(delta3, axis=0, keepdims=True) / m

    delta2 = np.dot(delta3, W3.T) * relu_derivative(z2)
    dW2 = np.dot(a1.T, delta2) / m
    db2 = np.sum(delta2, axis=0, keepdims=True) / m

    delta1 = np.dot(delta2, W2.T) * relu_derivative(z1)
    dW1 = np.dot(X.T, delta1) / m
    db1 = np.sum(delta1, axis=0, keepdims=True) / m

    return dW1, db1, dW2, db2, dW3, db3

