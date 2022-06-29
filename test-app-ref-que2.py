

#%%
import tensorflow as tf
print(tf.__version__)
import numpy as np
import cv2
from sklearn.preprocessing import StandardScaler
from time import sleep
from keras import models


texture_array_len = 18
texture_width = 100
texture_height = 100
#%%
classes = ["No gesture", "Swiping Left", "Swiping Right" , "Swiping Down", "Swiping Up"]
#%%
def rgb2gray(rgb):
    return np.dot(rgb[...,:3], [0.2989, 0.5870, 0.1140])

#%%
def normaliz_data(np_data):
    # Normalisation
    scaler = StandardScaler()
    #scaled_images  = normaliz_data2(np_data)
    scaled_images  = np_data.reshape(-1, texture_array_len, texture_width, texture_height, 3)
    return scaled_images

#%%
def normaliz_data2(v):
    normalized_v = v / np.sqrt(np.sum(v**2))
    return normalized_v


#%%

#%%
new_model = models.load_model('model-ver2-18frames-00024-0.90000.h5')

#%%
to_predict = []
num_frames = 0
cap = cv2.VideoCapture(0)
classe =''

while(True):
    # Capture frame-by-frame
    ret, frame = cap.read()

    # Our operations on the frame come here
    #gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    to_predict.append(cv2.resize(frame, (texture_width, texture_height)))
    
         
    if len(to_predict) == texture_array_len:
        frame_to_predict = np.array(to_predict, dtype=np.float32)
        frame_to_predict = normaliz_data(frame_to_predict)
        #print(frame_to_predict)
        predict = new_model.predict(frame_to_predict)
        classe = classes[np.argmax(predict)]
        print(predict)
        print('Classe = ',classe, 'Precision = ', np.amax(predict)*100,'%')


        #print(frame_to_predict)
        to_predict = []
        #sleep(0.1) # Time in seconds
        #font = cv2.FONT_HERSHEY_SIMPLEX
    cv2.putText(frame, classe, (30, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 0),1,cv2.LINE_AA)


    # Display the resulting frame
    cv2.imshow('Hand Gesture Recognition',frame)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break
cap.release()
cv2.destroyAllWindows()