import trainModel
import loadDataSet
from sklearn.model_selection import train_test_split
import os
import sys

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
#Dataset used is : https://console.cloud.google.com/storage/browser/quickdraw_dataset/full/raw;tab=objects?inv=1&invt=Ab4sfQ&prefix=&forceOnObjectsSortingFiltering=false
#Stored locally on computer but can be downloaded from the above link 
if __name__ == "__main__":
    # Set data folder path
    data_folder = r"C:\NumberRecognition\QuickDrawDoodles\Dataset\\"
    # Define your chosen categories (must match your downloaded .ndjson files)
    categories = [
        "apple",
        #"bicycle",
        #"cat",
        #"clock",
        #"dog",
        #"flower",
        #"moon",
        "pencil",
        #"rainbow",
        #"spoon",
        #"star",
        "sun",
        #"table",
        #"tree",
        #"umbrella"
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
    num_filters = 16  # Number of filters in the convolutional layer
    output_size = len(categories)  # match your number of classes dynamically
    output_size = len(categories)
    assert output_size > 1, f"Only {output_size} category found. Check your categories list!"

    learning_rate = 0.01
    epochs = 20  # Consider implementing early stopping
    batch_size = 256

    # Train your model
    trained_W_conv, trained_b_conv, trained_W1, trained_b1, trained_W2, trained_b2, trained_W3, trained_b3, trained_W4, trained_b4 = trainModel.train_model(
        x_train, y_train,
        x_test, y_test,
        input_size, hidden_size, hidden2_size, hidden3_size, output_size,
        epochs, learning_rate, batch_size,
        num_filters
    )

    print("\nModel training successful!")
