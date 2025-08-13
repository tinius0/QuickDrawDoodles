import model
import numpy as np
import visualizeTraining as vt

# Function to evaluate the model in batches
def evaluate_in_batches(X, y_one_hot, y_raw, W_conv, b_conv, W1, b1, W2, b2, W3, b3, W4, b4, batch_size=1024):
    outputs = []
    for i in range(0, len(X), batch_size):
        X_batch = X[i:i + batch_size]
        *_, output = model.forward_pass(X_batch, W_conv, b_conv, W1, b1, W2, b2, W3, b3, W4, b4, dropout_rate=0.0, training=False)
        outputs.append(output)
    outputs = np.vstack(outputs)
    loss = model.cross_entropy(y_one_hot, outputs)
    accuracy = np.mean(np.argmax(outputs, axis=1) == y_raw)
    return loss, accuracy


def train_model(X_train, y_train, X_test, y_test, input_size, hidden_size, hidden2_size, hidden3_size, output_size, epochs, initial_learning_rate, batch_size, num_filters):
    X_train = X_train.astype(np.float32)
    X_test = X_test.astype(np.float32)

    y_train_one_hot = model.one_hot_encode(y_train, output_size)
    y_test_one_hot = model.one_hot_encode(y_test, output_size)
    train_losses = []
    test_losses = []
    train_accuracies = []
    test_accuracies = []

    # Initialize parameters
    W_conv, b_conv, W1, b1, W2, b2, W3, b3, W4, b4 = model.init_params(input_size, hidden_size, hidden2_size, hidden3_size, output_size, num_filters)
    params = [W_conv, b_conv, W1, b1, W2, b2, W3, b3, W4, b4]

    m = [np.zeros_like(p) for p in params]
    v = [np.zeros_like(p) for p in params]
    beta1, beta2 = 0.9, 0.999
    eps = 1e-8
    weight_decay = 1e-5    
    t = 0

    num_training_samples = X_train.shape[0]

    print(f"Starting training with {num_training_samples} samples, {epochs} epochs, batch_size {batch_size}, initial_learning_rate {initial_learning_rate}")

    # compute train mean/std
    mean = X_train.mean(axis=0, keepdims=True)
    std = X_train.std(axis=0, keepdims=True) + 1e-8
    X_train = (X_train - mean) / std
    X_test  = (X_test  - mean) / std


    print("W4 shape:", W4.shape)
    print("b4 shape:", b4.shape)

    for epoch in range(epochs):
        permutation = np.random.permutation(num_training_samples)
        x_train_shuffled = X_train[permutation]
        y_train_shuffled_one_hot = y_train_one_hot[permutation]
        learning_rate = initial_learning_rate * (0.7 ** (epoch // 2))
        print(f"Epoch {epoch+1}/{epochs}, Learning Rate: {learning_rate:.6f}")

        for i in range(0, num_training_samples, batch_size):
            x_batch = x_train_shuffled[i:i + batch_size]
            y_batch_true = y_train_shuffled_one_hot[i:i + batch_size]

            # Forward pass batches
            X_reshaped, Z_conv, A_conv, A_pool, flattened, z1, a1, dropout_mask1, z2, a2, dropout_mask2, z3, a3, dropout_mask3, z4, output = model.forward_pass(x_batch, W_conv, b_conv, W1, b1, W2, b2, W3, b3, W4, b4, dropout_rate=0.3, training=True)
            
            # Calculate loss / backward propagation
            grads = model.backward_propagation(
                X_reshaped, Z_conv, A_conv, A_pool, flattened,
                z1, a1, dropout_mask1,
                z2, a2, dropout_mask2,
                z3, a3, dropout_mask3,
                z4, output,
                y_batch_true,
                W_conv, b_conv, W1, b1, W2, b2, W3, b3, W4, b4,
                dropout_rate=0.3
            )
            # Clip gradients
            clipped_gradients = model.clip_gradients(grads, clip_value=5.0)
            t += 1

            # Update parameters with Adam using clipped grads
            updated_params = []
            for idx in range(len(params)):
                g = clipped_gradients[idx]
                p = params[idx]
                
                m[idx] = beta1 * m[idx] + (1 - beta1) * g
                v[idx] = beta2 * v[idx] + (1 - beta2) * (g * g)
                
                m_hat = m[idx] / (1 - beta1 ** t)
                v_hat = v[idx] / (1 - beta2 ** t)
                
                updated_p = p - learning_rate * (m_hat / (np.sqrt(v_hat) + eps) + weight_decay * p)
                updated_params.append(updated_p)
            
            # Update the main parameters list with the new values
            W_conv, b_conv, W1, b1, W2, b2, W3, b3, W4, b4 = updated_params
            params = updated_params

        train_loss, train_accuracy = evaluate_in_batches(X_train, y_train_one_hot, y_train, W_conv, b_conv, W1, b1, W2, b2, W3, b3, W4, b4)
        test_loss, test_accuracy = evaluate_in_batches(X_test, y_test_one_hot, y_test, W_conv, b_conv, W1, b1, W2, b2, W3, b3, W4, b4)

        print(f"Epoch {epoch+1}, Train Loss: {train_loss:.4f}, Train Accuracy: {train_accuracy:.4f}, Test Loss: {test_loss:.4f}, Test Accuracy: {test_accuracy:.4f}")
        # Store losses and accuracies for visualization
        train_losses.append(train_loss)
        test_losses.append(test_loss)
        train_accuracies.append(train_accuracy)
        test_accuracies.append(test_accuracy)

    print("Training complete. Model saved to 'trained_model.pkl'.")
    vt.plot_training_progress(
        train_losses=train_losses,
        val_losses=test_losses,
        train_accuracies=train_accuracies,
        val_accuracies=test_accuracies
    )

    return W_conv, b_conv, W1, b1, W2, b2, W3, b3, W4, b4