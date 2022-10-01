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

# replace with your file
dataFrame = pd.read_csv(
    'week_39/winequalityN.csv')

head = dataFrame.head()  # see the first 5 rows

dataFrame.isnull().sum()  # will count number of rows
dataFrame.dropna(inplace=True)  # will remove rows with
# Thanks to David F. R. Petersen


pd.set_option('display.max_columns', None)  # print all columns
# select relevant rows and columns to X (here for example all rows and columns 5,6,7,8,9,10 and 11)
X = dataFrame.iloc[:, :-1]
# select column(s) for y (here all rows and only the last column)
y = dataFrame.iloc[:, -1]
# Capital X and lower-case y comes from Linear Algebra. The input is often a 2D array (matrix, named X) while the output is often a 1D array (vector, named y)

# convert ALL text-columns to categorical variables (One Hot encoding), e.g. gender, country etc.
X = pd.get_dummies(X)

print("------------------------", X.head())
# grab column-names before converting to numpy array
columnNames = list(X.columns)

print("head:", X.head())
X = X.values  # convert from Pandas dataframe to numpy array
y = y.values  # convert from Pandas dataframe to numpy array

scaler = StandardScaler()
# calculate mean and standard deviation and convert dataframe to numpy array
X = scaler.fit_transform(X)
# only use this, if the data is outside 0.0 … 1.0


X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42)

model = Sequential()
# 4 outputs. It will automatically adapt to number inputs
model.add(Dense(32, activation='relu'))
model.add(Dense(32, activation='relu'))
model.add(Dense(32, activation='relu'))
# Final output node for prediction. In this case, only one output neuron
model.add(Dense(units=1, activation='linear'))

# you may have to change learning_rate, if the model does not learn.
adam = Adam(learning_rate=0.001)
model.compile(loss='mse', optimizer=adam, metrics=['accuracy'])
# use loss = 'binary_crossentropy' for two-class classification.
# use loss = 'categorical_crossentropy' for multi-class classification.
# use loss = 'mse' (Mean Square Error) for regression (e.g. the Age,Height exercise).
# use metrics = ['accuracy']. It shows successful predictions / total predictions

# does the actual WORK !. verbose=1 will show output. 0 = no output.
model.fit(X_train, y_train, epochs=30, batch_size=10, verbose=1)

loss = model.history.history['loss']
sns.lineplot(x=range(len(loss)), y=loss)
plt.show()


print(model.evaluate(X_test, y_test, verbose=1))

y_pred = model.predict(X_test)
# creates a new array with true/false based on the boolean test
y_pred = (y_pred > 0.5)

cm = confusion_matrix(y_test, y_pred)
print(cm)

# will return a 2D array like this (random numbers):  jart@kea.dk
# [[6432   326]
# [ 481  1190]]

# interpretation:
# Top-left: 6432 correct predictions of 0.
# Top-right: 326 incorrect predictions of 1, when the y_test was 0.
# Bottom-left: 481 incorrect predictions of 0, when the y_test was 1.
# Bottom-right: 1190 correct predictions of 1

# first print column names, so you can enter new data in the correct columns
print(columnNames)
# enter new data in 2D array. Only numbers + dummy variables.

# enter new data in 2D array. Only numbers + dummy variables.
new_value = [[7, 0.27, 0.36, 20.7, 0.045,
              45, 170, 1.001, 3, 0.45, 8.8, 0, 1]]
new_value = scaler.transform(new_value)  # Don't forget to scale!
predict = model.predict(new_value)

print(predict)


model.save('my_model.h5')  # creates a HDF5 file 'my_model.h5'
# myModel is ready for predicting right away!