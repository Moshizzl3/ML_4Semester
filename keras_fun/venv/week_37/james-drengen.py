import numpy as np
import tensorflow as tf
from keras.models import Sequential
from keras.layers import Dense, Activation
from keras import optimizers

model = Sequential()
model.add(Dense(6, input_dim=6, activation='relu'))
model.add(Dense(6, activation='relu'))
model.add(Dense(3, activation='sigmoid'))
adam = optimizers.Adam(learning_rate=0.01)
model.compile(loss='mean_squared_error', optimizer=adam, metrics="accuracy")

x = np.array([[0.27, 0.24, 1, -1, 1, -1],
              [0.48, 0.98, -1, 1, -1, -1],
              [0.33, 0.44, -1, -1, -1, 1],
              [0.30, 0.29, 1, 1, -1, -1],
              [0.66, 0.65, -1, -1, 1, -1]
              ])
y = np.array([[0.43, 0.20, 0.37],
              [0.20, 0.43, 0.37],
              [0.20, 0.43, 0.37],
              [0.37, 0.20, 0.43],
              [0.43, 0.20, 0.37]])


class StopOnPoint(tf.keras.callbacks.Callback):
    def __init__(self, point):
        super(StopOnPoint, self).__init__()
        self.point = point

    def on_epoch_end(self, epoch, logs=None):
        loss = logs["loss"]
        if loss <= self.point:
            self.model.stop_training = True


callbacks = [StopOnPoint(1e-6)]  # <- set optimal point


history = model.fit(x, y, epochs=100000, batch_size=4, verbose=1, callbacks=callbacks)


prediction = model.predict([[0.38,0.51,-1,-1,1,-1]])

scores = model.evaluate(x, y)
print(prediction)
print('loss: ', scores[0])
print('acc: ', scores[1])
