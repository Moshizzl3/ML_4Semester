import numpy as np
import tensorflow as tf
from tensorflow import keras
from keras.models import Sequential
from keras.layers import Dense, Activation
from keras import optimizers

model = Sequential()
model.add(Dense(8, input_dim=1, activation='relu'))
model.add(Dense(6, input_dim=1, activation='relu'))
model.add(Dense(1, activation='linear'))
adam = optimizers.Adam(learning_rate=0.01)
model.compile(loss='mean_squared_error', optimizer=adam, metrics="accuracy")

x = np.array([[1], [3], [5], [7],[9],[11], [13]])
y = np.array([[75], [92], [108], [121], [130], [142], [155]])


class StopOnPoint(tf.keras.callbacks.Callback):
    def __init__(self, point):
        super(StopOnPoint, self).__init__()
        self.point = point

    def on_epoch_end(self, epoch, logs=None): 
        accuracy = logs["accuracy"]
        if accuracy >= self.point:
            self.model.stop_training = True

callbacks = [StopOnPoint(0.9999999)] # <- set optimal point


history = model.fit(x, y, epochs=10000, batch_size=8, verbose=1)


prediction = model.predict([[70]])

print(prediction)


