import cv2 as cv
import numpy as np
import  pickle #Used to load the trained model 


# Load the trained model
with open("ParametersAndTrainingData/trained_model.pkl", "rb") as f:
    W1, b1, W2, b2,W3,b3 = pickle.load(f)

#Functions from model.py to perform forward pass and prediction
def relu(x):
    return np.maximum(0,x)

def softmax(x):
    exp_x = np.exp(x - np.max(x, axis=1, keepdims=True))
    return exp_x / np.sum(exp_x, axis=1, keepdims=True)

def forward_pass(X,W1,b1,W2,b2,W3,b3):
    h1 = relu(np.dot(X, W1) + b1)
    h2 = relu(np.dot(h1, W2) + b2)
    out = np.dot(h2, W3) + b3  

    return np.argmax(out, axis=1)[0]

#Drawing interface for number recognition using greyscale images
img = np.zeros((280, 280), dtype=np.uint8)
drawing = False
last_point = None
WHITE = 255

def draw(event, x, y, flags, param):
    global drawing, last_point
    if event == cv.EVENT_LBUTTONDOWN:
        drawing = True
        last_point = (x, y)
    elif event == cv.EVENT_MOUSEMOVE and drawing:
        cv.line(img, last_point, (x, y), WHITE, thickness=24)
        last_point = (x, y)
    elif event == cv.EVENT_LBUTTONUP:
        drawing = False
        cv.line(img, last_point, (x, y), WHITE, thickness=24)

cv.namedWindow('Draw')
cv.setMouseCallback('Draw', draw)
prediction_text = ""  # Add this before your while loop

while True:
    display_img = img.copy()

    if prediction_text:
        cv.putText(display_img, f"Prediction: {prediction_text}", (10, 40), cv.FONT_HERSHEY_SIMPLEX, 1.2, (255), 3)
    cv.imshow('Draw', display_img)
    key = cv.waitKey(1) & 0xFF

    if key == ord('q'):
        break
    elif key == ord('c'):
        img = np.zeros((280, 280), dtype=np.uint8)
        prediction_text = ""
        
    elif key == ord('p'):
        resized_img = cv.resize(img, (28, 28), interpolation=cv.INTER_AREA)
        #processed = cv.bitwise_not(resized_img)
        coords = cv.findNonZero(resized_img)

        if coords is not None:
            x, y, w, h = cv.boundingRect(coords)
            digit = resized_img[y:y+h, x:x+w]
            # Make the digit fit in a 20x20 box
            digit = cv.resize(digit, (20, 20), interpolation=cv.INTER_AREA)

            padded = np.zeros((28, 28), dtype=np.uint8)
            x_offset = (28 - 20) // 2 #Centers the digit
            y_offset = (28 - 20) // 2
            padded[y_offset:y_offset+20, x_offset:x_offset+20] = digit
        else:
            padded = resized_img

        padded = padded.astype(np.float32) / 255.0
        padded = padded.reshape(1, 28 * 28)

        # Predict
        output = forward_pass(padded, W1, b1, W2, b2,W3,b3)
        #digit = np.argmax(output)
        prediction_text = str(output)

cv.destroyAllWindows()
