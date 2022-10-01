import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf
from keras.models import Sequential
from keras.layers import Dense, Activation
from keras import optimizers

model = Sequential()
model.add(Dense(3, input_dim=1, activation='relu', use_bias=True, bias_initializer='ones'))
model.add(Dense(units=1, activation='linear'))
adam = optimizers.Adam(learning_rate=0.01)
model.compile(loss='mean_squared_error', optimizer=adam, metrics="accuracy")

x = np.array([[1], [3], [5], [7],[9],[11], [13]])
y = np.array([[75], [92], [108], [121], [130], [142], [155]])

history = model.fit(x, y, epochs=2000, batch_size=2, verbose=2)


prediction = model.predict([[70]])

scores = model.evaluate(x, y)
print(prediction)
print('loss: ', scores[0])
print('acc: ', scores[1])

plt.plot(x,y)
plt.show()



