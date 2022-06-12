import numpy as np
import cv2
from PIL import Image
import tensorflow as tf
import os

abspath = os.path.abspath(__file__)
dname = os.path.dirname(abspath)
os.chdir(dname)
cam = cv2.VideoCapture(0)
model = tf.keras.models.load_model("model.h5")
class_names = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9]
def predict_number(prediction_image):
    global model, class_names
    predictions = model.predict(prediction_image)
    #print(predictions)
    prediction_final = np.argmax(predictions)
    #print(prediction_final)
    #print(f"Prediction: {class_names[prediction_final]}")
    return class_names[prediction_final]

while True:
    ret, win = cam.read()
    window_resized = win[0:450, 0:350]
    cv2.rectangle(win, (100, 100), (250, 250), (255, 0, 0), 0)
    img = win[100:250, 100:250]
    im_resized = cv2.resize(img, (28, 28), Image.ANTIALIAS)
    im_resized_array = np.asarray(im_resized)
    im_resized_array = im_resized_array[:,:,0]
    im_resized_array = im_resized_array / 255
    final_image = np.reshape(im_resized_array, (1, 28, 28, 1))
    prediction = predict_number(final_image)
    image = cv2.putText(window_resized, str(prediction), (100, 100), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2, cv2.LINE_AA) 
    cv2.imshow("Camera", window_resized)
    if cv2.waitKey(1) == ord(' '):
        break
cam.release()
cv2.destroyAllWindows()