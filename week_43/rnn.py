import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import tensorflow as tf
import datetime
from sklearn.preprocessing import MinMaxScaler
from numpy.core.fromnumeric import shape
from keras.models import Sequential
from keras.layers import Dense, LSTM, Dropout
from keras.optimizers import Adam

# Set constants. This makes it easier to adjust for future problems.

# parameters
daysBack = 240  # how many days to look back, when predicting price movement
lookAhead = 40  # predict price, after this many days
dropRate = 0.2  # drop 20% of neurons to prevent overfitting
units_ = 120  # number of neurons
learningRate = 0.001
batchSize = 32
epochs_ = 50

# Get training data from file. Convert from Pandas datafram to Numpy array.
# import training set.
dataset_train = pd.read_csv(
    'week_43/googleStockprice/Google_Stock_Price_Train.csv')
training_set = dataset_train.iloc[:, 1:2].values  # return numpy arr

# Scale data using normalization
# scales input to values between 0.0 and 1.0
sc = MinMaxScaler(feature_range=(0, 1))
training_set_scaled = sc.fit_transform(training_set)

# Create X_train and y_train from the scaled data in previous step
# Create a datastructure with x number of timesteps and 1 output.
# Important question: how many days to look back ? 10,20... ?

X_train = []  # pyton list
y_train = []
for i in range(daysBack, len(training_set) - lookAhead):
    X_train.append(training_set_scaled[i - daysBack:i, 0])
    if training_set[i + lookAhead, 0] - training_set[i, 0] > 0:
        y_train.append(1)  # price is higher
    else:
        y_train.append(0)  # price is lower
# X_train and y_train are lists. Need to be cast to numpy arrays
X_train, y_train = np.array(X_train), np.array(y_train)

# Reshape array from 2D to 3D, to enable other data to be added later, if necessary

# add one more dimension, so we go from
X_train = np.reshape(X_train, (X_train.shape[0], X_train.shape[1], 1))
# (rows_, cols_) to (rows_, cols_, 1). The last 1 can be replaced with 2,3,4 etc. if you have more data,
# f.x. trading volume, high/low etc.

# Create model
model = Sequential()

# add the first LSTM layer with dropout regularization
model.add(LSTM(dropout=dropRate, units=units_,
          return_sequences=True, input_shape=(X_train.shape[1], 1)))
# input_shape is necessary, because we will have a certain number of columns of training data for
# each row. Amount of rows is given.
# use return_sequences to indicate, that another LSTM is coming afterwards
# dropout=dropRate for each iteration, turn off a part (f.x. 20%) of the neurons from current layer.
# It makes the model more sparse (less dense), which forces it to learn even without some neurons.
# This in turn makes the model more robust when predicting unseen new data!

model.add(LSTM(dropout=dropRate, units=units_,
          return_sequences=True))  # 2nd layer
model.add(LSTM(dropout=dropRate, units=units_,
          return_sequences=True))  # 3rd layer
model.add(LSTM(dropout=dropRate, units=units_,
          return_sequences=True))  # 4th layer
# 5th layer, return_sequences=False, which is required
model.add(LSTM(dropout=dropRate, units=units_, return_sequences=False))

model.add(Dense(1, activation="sigmoid"))

adam = Adam(learning_rate=learningRate)
model.compile(optimizer=adam, loss="binary_crossentropy", metrics=['accuracy'])

log_dir = "./week_43/logs/" + datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
tensorboard_callback = tf.keras.callbacks.TensorBoard(
    log_dir=log_dir, histogram_freq=1)


# Train
#model.fit(X_train, y_train, batch_size=batchSize, epochs=epochs_,
          #verbose=1, callbacks=tensorboard_callback)

# you should get accuracy > 50%. For the Google Stock Price I have reached ~72% accuracy

# Prepare X_test array for evaluation
# here you have to get new data, which the model has not seen before. Download test data from here.
dataset_test = pd.read_csv(
    'week_43/googleStockprice/Google_Stock_Price_Test.csv')  # 20 days to be tested
# axis=0 is Stack on top
dataset_total = pd.concat(
    (dataset_train['Open'], dataset_test['Open']), axis=0)
# next, we grab the last 300 entries from dataset_total: len(dataset_test) = 20, daysBack = 240, lookAhead = 40.   20 + 240 + 40 = 300.
inputs = dataset_total[len(dataset_total) -
                       len(dataset_test) - daysBack - lookAhead:].values
# -1 means that numpy should figure out the shape. Here we reshape from 1-d to
inputs = inputs.reshape(-1, 1)
# a 2-d shape. (-1,1) means: Numpy should figure out, how many rows. But set number of columns to 1.
# transform to values between 0.0 and 1.0. Keep the inputs array, to compare real prices,
inputsScaled = sc.transform(inputs)
# when creating the y_test array.

X_test = []  # python list
y_test = []
for i in range(0, len(dataset_test)):  # iterate over 20 financial days
    X_test.append(inputsScaled[i: i + daysBack, 0])
    if inputs[i + daysBack + lookAhead, 0] - inputs[i + daysBack, 0] > 0:
        y_test.append(1)
    else:
        y_test.append(0)

# X_train and y_train are python lists. Need to be cast to numpy arrays
X_test = np.array(X_test)  # convert from python list to numpy array
y_test = np.array(y_test)
# reshape from 2-d to 3-d numpy array.
X_test = np.reshape(X_test, (X_test.shape[0], X_test.shape[1], 1))

model.fit(X_train, y_train, batch_size=batchSize, epochs=epochs_,
          verbose=1, callbacks=tensorboard_callback, validation_data=(X_test, y_test))

# Evaluate model
model.evaluate(X_test, y_test, verbose=1)  # will return percentage

# Predict a new single value
# for example predict the price direction, based on the first row of the X_test array
predictedPriceDirection = model.predict(X_test[0:1])
# result > 0.5 means price will be higher, and vice versa
print(predictedPriceDirection)
