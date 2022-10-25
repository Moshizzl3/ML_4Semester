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

myVegieDic = {0: 'Bean',  1: 'Bitter_Gourd',  2: 'Bottle_Gourd',  3: 'Brinjal', 4: 'Broccoli',
              5: 'Cabbage', 6: 'Capsicum',  7: 'Carrot',
              8: 'Cauliflower',  9: 'Cucumber', 10: 'Papaya', 11: 'Potato',
              12: 'Pumpkin', 13: 'Radish',  14: 'Tomato'}

targetSize = 62
color = 'rgb'

#scaler = joblib.load('scaler.gz')
myModel = load_model('model_vegies1.h5')

#singlePred = 'preds/carrots3.jpg'
#singlePred = 'preds/pat.jpg'
#singlePred = 'preds/cucumbers1.jpg'
singlePred = 'preds/califlower1.jpeg'
#singlePred = 'week_41/vegies/validation/Bottle_Gourd/1204.jpg'
test_image = image.image_utils.load_img(
    singlePred, target_size=[targetSize, targetSize], color_mode=color)
# here set the SAME parameters as on the training in Step 5.
# possible solution: image.image_utils.load_img  Set image_utils after image.

test_image = image.image_utils.img_to_array(
    test_image)  # convert image to array
# add one extra dimension to hold batch.
test_image = np.expand_dims(test_image, axis=0)
# axis=0 means that a new dimension will be added, such that test_image.shape goes from (28, 28, 1) to (1, 28, 28, 1). This is required by Tensorflow.

# remember to divide each pixel value by 255.0
result = myModel.predict(test_image/255.0)
result = np.array(result)

print("result is: " + str(np.argmax(result)))

count = 0
for i in result[0]:
    print(count,":",myVegieDic[count],"---", i*100,)
    count += 1
