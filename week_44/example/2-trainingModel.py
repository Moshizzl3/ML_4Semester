from keras.models import Sequential
from keras.layers import Dense
from keras.models import load_model
import numpy as np
from keras.optimizers import Adam
from pathlib import Path

trainingData = np.load('saved.npy', allow_pickle=True)
inputData = trainingData[:, 0].tolist()
outputData = trainingData[:, 1].tolist()
my_file = Path("gamemodel.h5")
if my_file.is_file():
   model = load_model('gamemodel.h5') # loads pre-trained model
   print('Model found')
else:
   model = Sequential() # creates new, empty model
   model.add(Dense(10, input_dim=4, activation='relu'))
   model.add(Dense(64, activation='relu'))
   #model.add(Dense(64, activation='tanh')) # can be used…
   model.add(Dense(2, activation='softmax'))
   model.compile(loss='categorical_crossentropy', optimizer=Adam(lr=0.0001), metrics=['accuracy'])
   print('New model created, no pre-trained model found...')

print(model.input)
model.fit(inputData, outputData, verbose=1, epochs=20)
model.save('gamemodel.h5')
