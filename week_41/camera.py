import cv2
from operator import concat
import numpy as np
import pandas as pd
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from keras.models import Sequential, load_model
from keras.layers import Dense
from keras.optimizers import Adam
from sklearn.metrics import confusion_matrix, accuracy_score
import matplotlib.pyplot as plt
from keras.utils import load_img, img_to_array
from keras.preprocessing import image
from PIL import Image

vc = cv2.VideoCapture(0)


myVegieDic = {'Bean': 0, 'Bitter_Gourd': 1, 'Bottle_Gourd': 2, 'Brinjal': 3, 'Broccoli': 4, 
              'Cabbage': 5, 'Capsicum': 6, 'Carrot': 7, 
              'Cauliflower': 8, 'Cucumber': 9, 'Papaya': 10, 'Potato': 11, 
              'Pumpkin': 12, 'Radish': 13, 'Tomato': 14}


while True:

    ret, frame = vc.read()

    targetSize = 62
    color = 'rgb'

    #scaler = joblib.load('scaler.gz')q
    myModel = load_model('model_vegies1.h5')

    singlePred = Image.fromarray(frame)
    print(singlePred)
    singlePred = singlePred.resize((targetSize,targetSize))
    print(singlePred)
    singlePred = np.expand_dims(singlePred, axis=0)
    # axis=0 means that a new dimension will be added, such that test_image.shape goes from (28, 28, 1) to (1, 28, 28, 1). This is required by Tensorflow.

    # remember to divide each pixel value by 255.0
    result = myModel.predict(singlePred/255.0)
    result = np.array(result)


    print(result)
    print(np.argmax(result))
    resultIndex = np.argmax(result)


    cv2.putText(frame, 'It is a: ' + (list(myVegieDic.keys())
      [list(myVegieDic.values()).index(resultIndex)]), (10,100),
                cv2.FONT_HERSHEY_SIMPLEX,
                4,(50,100,0), 4, 2)
    cv2.imshow('Camera feed', frame)
  

    if cv2.waitKey(20) & 0xFF == ord('q'):
      break
    
vc.release()
# Destroy all the windows
cv2.destroyAllWindows()