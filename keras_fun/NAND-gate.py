import numpy as np
import tensorflow as tf
from tensorflow import keras
from keras.models import Sequential
from keras.layers import Dense, Activation
from keras import optimizers

model = Sequential()
model.add(Dense(8, input_dim=2, activation='relu'))
model.add(Dense(1, activation='sigmoid'))
adam = optimizers.Adam(learning_rate=0.01)
model.compile(loss='mean_squared_error', optimizer=adam, metrics="accuracy")

x = np.array([[0, 0], [1, 0], [0, 1], [1, 1]])
y = np.array([[1], [1], [1], [0]])


class StopOnPoint(tf.keras.callbacks.Callback):
    def __init__(self, point):
        super(StopOnPoint, self).__init__()
        self.point = point

    def on_epoch_end(self, epoch, logs=None):
        accuracy = logs["accuracy"]
        if accuracy >= self.point:
            self.model.stop_training = True


callbacks = [StopOnPoint(0.9999999)]  # <- set optimal point


history = model.fit(x, y, epochs=10000, batch_size=2, verbose=1)


prediction = model.predict([[1, 1]])
scores = model.evaluate(x, y)
print(prediction)
