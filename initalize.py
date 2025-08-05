from . import trainModel
from . import loadDataSet
from sklearn.model_selection import train_test_split
import os
import sys

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

if __name__ == "__main__":
    # Set your data folder path
    data_folder = "C:\\Users\\tiniu\\Documents\\NumberRecognition\\QuickDrawDoodles\\DataSet"
    # Define your chosen categories (must match your downloaded .ndjson files)
    categories = [
    "airplane",
    "apple",
    "banana",
    "bicycle",
    "car",
    "cat",
    "clock",
    "dog",
    "flower",
    "house",
    "moon",
    "motorbike",
    "pencil",
    "rainbow",
    "school bus",
    "snowman",
    "spoon",
    "star",
    "sun",
    "table",
    "television",
    "train",
    "tree",
    "truck",
    "umbrella",
    "whale",
    "wheel"
]

    # Load the dataset (pass both folder and categories)
    x, y = loadDataSet.load_quickdraw_ndjson(data_folder, categories)

    # Split train/test
    x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=42)

    # Model parameters
    input_size = 28 * 28
    hidden_size = 512
    hidden2_size = 256
    hidden3_size = 128
    output_size = len(categories)  # match your number of classes dynamically

    learning_rate = 0.05
    epochs = 10  # Consider implementing early stopping
    batch_size = 128

    # Train your model
    trained_W1, trained_b1, trained_W2, trained_b2, trained_W3, trained_b3, trained_W4, trained_b4 = trainModel.train_model(
        x_train, y_train,
        x_test, y_test,
        input_size, hidden_size, hidden2_size, hidden3_size, output_size,
        epochs, learning_rate, batch_size
    )

    print("\nModel training successful!")

