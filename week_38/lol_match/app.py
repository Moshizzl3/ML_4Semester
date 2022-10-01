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
import joblib

scaler = joblib.load('scaler.gz')
myModel = load_model('my_model.h5')

new_value = [[25032,311,64.50000011920929,9,36146,351,61.50000008940697,10.4,9,0,1,0,24,1,1,4]]
new_value = scaler.transform(new_value)  # Don't forget to scale!
predict = myModel.predict(new_value)

print(predict)