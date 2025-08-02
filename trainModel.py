import model
import numpy as np
import pickle  # Used to save the trained model
import visualizeTraining as vt

def train_model(X_train, y_train, X_test, y_test, input_size, hidden_size,hidden2_size, output_size, epochs, learning_rate, batch_size):
    y_train_one_hot = model.one_hot_encode(y_train, output_size)
    y_test_one_hot = model.one_hot_encode(y_test, output_size)
    train_losses = []
    test_losses = []
    train_accuracies = []
    test_accuracies = []

    #Initialize parameters
    W1,b1,W2,b2,W3,b3 = model.init_params(input_size, hidden_size,hidden2_size, output_size)

    num_training_samples = X_train.shape[0]

    print(f"Starting training with {num_training_samples} samples, {epochs} epochs, batch_size {batch_size}, learning_rate {learning_rate}")

    for epoch in range(epochs):
        permutation = np.random.permutation(num_training_samples)
        x_train_shuffled = X_train[permutation]
        y_train_shuffled_one_hot = y_train_one_hot[permutation]


        for i in range(0, num_training_samples, batch_size):
            x_batch = x_train_shuffled[i:i + batch_size]
            y_batch_true = y_train_shuffled_one_hot[i:i + batch_size]

            #Forward pass batches
            z1, a1, z2, a2, z3, output = model.forward_pass(x_batch, W1, b1, W2, b2, W3, b3)

            #Calculate loss / backward propagation
            dW1, db1, dW2, db2,dW3,db3 = model.backward_propagatation(x_batch, y_batch_true,z1, a1, z2, a2,z3, output, W2,W3)

            #Update the weights and biases
            W1 -= learning_rate * dW1
            b1 -= learning_rate * db1
            W2 -= learning_rate * dW2
            b2 -= learning_rate * db2
            b3 -= learning_rate * db3
            W3 -= learning_rate * dW3

        _,_,_,_,_, train_output = model.forward_pass(X_train, W1, b1, W2, b2, W3, b3)
        train_loss = model.cross_entropy(y_train_one_hot, train_output)
        train_accuracy = np.mean(np.argmax(train_output, axis=1) == y_train)


        _,_,_,_,_, test_output = model.forward_pass(X_test, W1, b1, W2, b2, W3, b3)
        test_loss = model.cross_entropy(y_test_one_hot, test_output)
        test_accuracy = np.mean(np.argmax(test_output, axis=1) == y_test)

        print(f"Epoch {epoch+1}, Batch {i//batch_size+1}, Train Loss: {train_loss:.4f}, Train Accuracy: {train_accuracy:.4f}, Test Loss: {test_loss:.4f}, Test Accuracy: {test_accuracy:.4f}")
        # Store losses and accuracies for visualization
        train_losses.append(train_loss)
        test_losses.append(test_loss)
        train_accuracies.append(train_accuracy)
        test_accuracies.append(test_accuracy)

        # Save trained weights to file
        with open("ParametersAndTrainingData/trained_model.pkl", "wb") as f:
            pickle.dump((W1, b1, W2, b2,W3,b3), f)

    print("Training complete. Model saved to 'trained_model.pkl'.")
    vt.plot_training_progress(
        train_losses=train_losses,
        val_losses=test_losses,
        train_accuracies=train_accuracies,
        val_accuracies=test_accuracies
    )

    return W1, b1, W2, b2, W3, b3