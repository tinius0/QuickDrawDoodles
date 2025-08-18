import trainModel
import loadDataSet
from sklearn.model_selection import train_test_split
import os
import sys

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
# Dataset used is: https://console.cloud.google.com/storage/browser/quickdraw_dataset/full/raw
# Stored locally on computer but can be downloaded from the link above  

if __name__ == "__main__":
    # Set data folder path
    data_folder = r"C:\NumberRecognition\QuickDrawDoodles\Dataset\\"

    # Define your chosen categories (must match your downloaded .ndjson files)
    categories = [
        "apple",
        #"pencil",
        #"sun",
        #"umbrella",
        #"tree"
    ]

    # Load the dataset (pass both folder and categories)
    x, y = loadDataSet.load_quickdraw_ndjson(data_folder, categories)

    # Split train/test
    x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=42)

    # Model parameters
    input_shape = (1,28, 28) 
    hidden_size = 1024
    hidden2_size = 521
    hidden3_size = 256
    num_filters1 = 8
    num_filters2 = 16
    num_filters3 = 32
    output_size = len(categories)
    assert output_size > 0, f"Only {output_size} category found. Check your categories list!"

    learning_rate = 0.001
    epochs = 20  # Consider implementing early stopping
    batch_size = 32

    # Train your model
    trained_params = trainModel.train_model(
        x_train, y_train,
        x_test, y_test,
        input_shape, hidden_size, hidden2_size, hidden3_size, output_size,
        epochs, learning_rate, batch_size,
        num_filters1, num_filters2, num_filters3
    )

    print("\nModel training successful!")
