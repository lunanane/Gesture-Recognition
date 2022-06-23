import cv2
import numpy as np
from PIL import Image
from keras import models

#Load the saved model
model = models.load_model('model-00035-0.79167.h5')
video = cv2.VideoCapture(0)

class_names = ["No gesture", "Swiping Left", "Swiping Right" , "Swiping Down", "Swiping Up"]
while True:
    _, frame = video.read()

    #Convert the captured frame into RGB
    im = Image.fromarray(frame, 'RGB')
    #Resizing into 128x128 because we trained the model with this image size.
    im = im.resize((100,100))
    
    
    img_array = np.array(im)

    img_array = np.expand_dims(img_array, axis=0)

    #img_array = img_array.reshape( (1,) + img_array.shape )
    #Our keras model used a 4D tensor, (images x height x width x channel)
    #So changing dimension 128x128x3 into 1x128x128x3 

    img_array = np.expand_dims(img_array, axis=0)




    #print(img_array)
    #Calling the predict method on model to predict 'me' on the image
    #what is the difference between predict and predict_on_batch??

    #i = model.predict(img_array)[0][0]
    i = model.predict(img_array)[0]
    
    prediction_final = np.argmax(i)
    #preds = model.predict(img_array)[0]
    #Q.append(preds)
    
    #results = np.array(Q).mean(axis=0)
    #i = np.argmax(results)
    
    print(i)
    print (prediction_final)
    #if prediction is 0, which means I am missing on the image, then show the frame in gray color.
    #if prediction == 0:
    
        #frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    cv2.imshow("Capturing", frame)
    key=cv2.waitKey(1)

    if key == ord('q'):
        break
        
video.release()
cv2.destroyAllWindows()