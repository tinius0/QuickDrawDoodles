from flask import Flask, request, jsonify, render_template
from PIL import Image
import numpy as np
import pickle
import sys
import os

app = Flask(__name__)
# This line is added to handle path issues with your project structure
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

# Define the categories you are actually using for prediction
categories = [
    "apple",
    "pencil",
    "sun"
]

# Load the tuple of weights and biases from the pickle file
try:
    with open("trained_model.pkl", "rb") as f:
        # We need to unpack the tuple into individual parameters
        W1, b1, W2, b2, W3, b3, W4, b4 = pickle.load(f)
except Exception as e:
    print(f"Error loading model: {e}")
    W1, b1, W2, b2, W3, b3, W4, b4 = None, None, None, None, None, None, None, None


# --- Helper Functions for the Forward Pass ---
def relu(x):
    """ReLU activation function."""
    return np.maximum(0, x)

def softmax(x):
    """Softmax activation function for the output layer."""
    exp_x = np.exp(x - np.max(x, axis=1, keepdims=True))
    return exp_x / np.sum(exp_x, axis=1, keepdims=True)

# --- End Helper Functions ---


@app.route("/")
def index():
    return render_template("canvas.html")

@app.route("/predict", methods=["POST"])
def predict():
    # Check if the model parameters were loaded successfully
    if W1 is None:
        return jsonify({"prediction": "Model Load Error", "confidence": 0.0}), 500

    try:
        # Get the image file from the request
        file = request.files["image"]
        
        # Process the image to match the model's input
        img = Image.open(file).convert("L")  # Convert to grayscale
        img = img.resize((28, 28))             # Resize to 28x28 pixels
        img_array = np.array(img) / 255.0      # Normalize pixel values
        img_array = img_array.flatten().reshape(1, -1) # Flatten to a 1D array

        # --- Manual Forward Pass ---
        # Layer 1
        z1 = np.dot(img_array, W1) + b1
        a1 = relu(z1)
        
        # Layer 2
        z2 = np.dot(a1, W2) + b2
        a2 = relu(z2)
        
        # Layer 3
        z3 = np.dot(a2, W3) + b3
        a3 = relu(z3)
        
        # Output Layer
        z4 = np.dot(a3, W4) + b4
        output = softmax(z4)
        # --- End Manual Forward Pass ---

        # The prediction is the index of the highest probability
        prediction_index = np.argmax(output, axis=1)[0]
        
        # The confidence is the highest probability itself
        confidence = np.max(output)

        # Check if the prediction_index is a valid index for our categories list
        if prediction_index < len(categories):
            prediction = categories[prediction_index]
        else:
            # If the index is out of range, we return a generic "unknown" prediction.
            # This confirms that the model's output is not limited to the first three categories.
            prediction = "Unknown"
        
        # Return the actual prediction and confidence as a JSON response
        return jsonify({"prediction": prediction, "confidence": confidence})

    except Exception as e:
        print(f"An error occurred during prediction: {e}")
        return jsonify({"prediction": "Error", "confidence": 0.0}), 500

if __name__ == "__main__":
    app.run(debug=True)