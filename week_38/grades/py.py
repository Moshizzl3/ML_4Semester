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

# replace with your file
dataFrame = pd.read_csv(
    'week_38/grades/student_prediction.csv')

head = dataFrame.head()  # see the first 5 rows

print(head)

dataFrame.isnull().sum()  # will count number of rows
dataFrame.dropna(inplace=True)  # will remove rows with
# Thanks to David F. R. Petersen

pd.set_option('display.max_columns', None)  # print all columns
# select relevant rows and columns to X (here for example all rows and columns 5,6,7,8,9,10 and 11)
X = dataFrame.iloc[:, 1:-1]
# select column(s) for y (here all rows and only the last column)
y = dataFrame.iloc[:, -1]
# Capital X and lower-case y comes from Linear Algebra. The input is often a 2D array (matrix, named X) while the output is often a 1D array (vector, named y)

# convert ALL text-columns to categorical variables (One Hot encoding), e.g. gender, country etc.
X = pd.get_dummies(X)

print("------------------------",X.head())
# grab column-names before converting to numpy array
columnNames = list(X.columns)

print("head:", X.head())
X = X.values  # convert from Pandas dataframe to numpy array
y = y.values  # convert from Pandas dataframe to numpy array

scaler = StandardScaler()
# calculate mean and standard deviation and convert dataframe to numpy array
X = scaler.fit_transform(X)
# only use this, if the data is outside 0.0 … 1.0


y = y.astype(np.int8) # convert to a type, dunno why? nothing seems to change if u print value before/after

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42)

model = Sequential()
# 4 outputs. It will automatically adapt to number inputs
model.add(Dense(116, activation='relu'))
model.add(Dense(116, activation='relu'))
# Final output node for prediction. In this case, only one output neuron
model.add(Dense(1, activation='relu'))

# you may have to change learning_rate, if the model does not learn.
adam = Adam(learning_rate=0.001)
model.compile(loss='mse', optimizer=adam, metrics=['accuracy'])
# use loss = 'binary_crossentropy' for two-class classification.
# use loss = 'categorical_crossentropy' for multi-class classification.
# use loss = 'mse' (Mean Square Error) for regression (e.g. the Age,Height exercise).
# use metrics = ['accuracy']. It shows successful predictions / total predictions

# does the actual WORK !. verbose=1 will show output. 0 = no output.
model.fit(X_train, y_train, epochs=100, batch_size=4, verbose=1)

loss = model.history.history['loss']
sns.lineplot(x=range(len(loss)), y=loss)

model.evaluate(X_test, y_test, verbose=1)

y_pred = model.predict(X_test)
y_pred = y_pred.flatten()
print(y_pred.shape)
print(y_test.shape)
print(type(y_pred))
print(type(y_test))
compareArr = np.vstack((y_pred, y_test)).T
print(compareArr)
# Saving the array in a text file
np.savetxt("activation.txt", compareArr)

content = np.loadtxt('activation.txt')

print(type(content))
sum = 0
for x in range(len(content)):
  diff = abs(content[x][1] - content[x][0])
  print(content[x][1], "-", content[x][0], " = ", diff)
  sum += diff
print(sum / len(content))

print(columnNames) # first print column names, so you can enter new data in the correct columns
new_value = [[2,1,2,4,2,1,2,1,1,2,3,3,2,3,2,5,5,2,2,1,1,1,1,2,2,2,3,1,3,3,8]] # enter new data in 2D array. Only numbers + dummy variables. 
new_value = scaler.transform(new_value) # Don't forget to scale! 5
prediction = model.predict(new_value)

print(prediction)
model.save('my_model.h5')  # creates a HDF5 file 'my_model.h5'

myModel = load_model('my_model.h5') # myModel is ready for predicting right away!
