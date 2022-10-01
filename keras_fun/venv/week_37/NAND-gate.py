import numpy as np
import tensorflow as tf
from tensorflow import keras
from keras.models import Sequential
from keras.layers import Dense, Activation
from keras import optimizers
from keras import initializers

model = Sequential()
model.add(Dense(3, input_dim=2, activation='relu', use_bias=True, bias_initializer= "ones"))
model.add(Dense(1, activation='sigmoid'))
adam = optimizers.Adam(learning_rate=0.01)
model.compile(loss='mean_squared_error', optimizer=adam, metrics="accuracy")

x = np.array([[0, 0], [1, 0], [0, 1], [1, 1]])
y = np.array([[0], [0], [0], [1]])


class StopOnPoint(tf.keras.callbacks.Callback):
    def __init__(self, point):
        super(StopOnPoint, self).__init__()
        self.point = point

    def on_epoch_end(self, epoch, logs=None):
        loss = logs["loss"]
        if loss <= self.point:
            self.model.stop_training = True


callbacks = [StopOnPoint(1e-3)]  # <- set optimal point


history = model.fit(x, y, epochs=100000, batch_size=2, verbose=1, callbacks=callbacks)


prediction = model.predict([[1, 1]])
scores = model.evaluate(x, y)
print(prediction)
print('loss: ', scores[0])
print('acc: ', scores[1])