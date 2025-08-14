import numpy as np
import model
import visualizeTraining as vt

# Same for evaluation
def evaluate_in_batches(X, y_one_hot, y_raw, params, batch_size=256):
    outputs = []
    for i in range(0, len(X), batch_size):
        X_batch = X[i:i + batch_size].reshape(-1, 1, 28, 28)
        caches = model.forward_pass(X_batch, params, dropout_rate=0.1)
        outputs.append(caches['A4'])
    outputs = np.vstack(outputs)
    loss = model.cross_entropy(y_one_hot, outputs)
    accuracy = np.mean(np.argmax(outputs, axis=1) == y_raw)
    return loss, accuracy


# Follow up on input_shape change
def train_model(X_train, y_train, X_test, y_test,
                input_size, hidden_size, hidden2_size, hidden3_size, output_size,
                epochs, initial_learning_rate, batch_size,
                num_filters1, num_filters2, num_filters3):

    X_train = X_train.astype(np.float32)
    X_test = X_test.astype(np.float32)
    y_train_one_hot = model.one_hot_encode(y_train, output_size)
    y_test_one_hot = model.one_hot_encode(y_test, output_size)

    train_losses, test_losses = [], []
    train_accuracies, test_accuracies = [], []

    # Initialize parameters (now with 3 conv layers)
    params = model.init_params(
    input_shape=(1, 28, 28),
    conv_filters=[num_filters1, num_filters2, num_filters3],
    hidden_sizes=[hidden_size, hidden2_size, hidden3_size],
    output_size=output_size,
    kernel_size=3
    )


    # Adam states
    m = {k: np.zeros_like(v) for k, v in params.items()}
    v = {k: np.zeros_like(v) for k, v in params.items()}
    beta1, beta2 = 0.9, 0.999
    eps = 1e-8
    weight_decay = 1e-5
    t = 0

    # Normalize
    mean = X_train.mean(axis=0, keepdims=True)
    std = X_train.std(axis=0, keepdims=True) + 1e-8
    X_train = (X_train - mean) / std
    X_test = (X_test - mean) / std

    num_samples = X_train.shape[0]
    print(f"Starting training with {num_samples} samples, {epochs} epochs, batch_size {batch_size}")

    for epoch in range(epochs):
        permutation = np.random.permutation(num_samples)
        X_shuffled = X_train[permutation]
        y_shuffled = y_train_one_hot[permutation]

        for i in range(0, num_samples, batch_size):
            X_batch = X_shuffled[i:i + batch_size].reshape(-1, 1, 28, 28)
            y_batch = y_shuffled[i:i + batch_size]

            # Forward pass
            caches = model.forward_pass(X_batch, params)
            # Backward pass
            grads = model.backward_pass(caches, params, y_batch)
            grads = model.clip_gradients(grads, 5.0)

            t += 1
            # Adam update of parameters
            for key in params.keys():
                g = grads[key]
                m[key] = beta1 * m[key] + (1 - beta1) * g
                v[key] = beta2 * v[key] + (1 - beta2) * (g * g)
                m_hat = m[key] / (1 - beta1 ** t)
                v_hat = v[key] / (1 - beta2 ** t)
                params[key] -= initial_learning_rate * (m_hat / (np.sqrt(v_hat) + eps) + weight_decay * params[key])

        # Evaluate after each epoch
        train_loss, train_acc = evaluate_in_batches(X_train, y_train_one_hot, y_train, params)
        test_loss, test_acc = evaluate_in_batches(X_test, y_test_one_hot, y_test, params)

        train_losses.append(train_loss)
        test_losses.append(test_loss)
        train_accuracies.append(train_acc)
        test_accuracies.append(test_acc)

        print(f"Epoch {epoch+1}/{epochs} | Train Loss: {train_loss:.4f}, Acc: {train_acc:.4f} | Test Loss: {test_loss:.4f}, Acc: {test_acc:.4f}")

    # Plot training progress
    vt.plot_training_progress(train_losses, test_losses, train_accuracies, test_accuracies)

    return params
