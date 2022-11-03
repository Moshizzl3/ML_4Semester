from keras.models import Sequential
from keras.layers import Dense, Dropout
from keras.models import load_model
import numpy as np
from keras.optimizers import Adam
from pathlib import Path

trainingData = np.load('saved.npy', allow_pickle=True)
inputData = trainingData[:, 0].tolist()
outputData = trainingData[:, 1].tolist()
print(len(trainingData))
# my_file = Path("gamemodel.h5")
# if my_file.is_file():
#    model = load_model('gamemodel.h5') # loads pre-trained model
#    print('Model found')
# else:
#       model = Sequential()  # creates new, empty model
#       model.add(Dense(128, input_dim=4, activation='relu'))
#       model.add(Dropout(0.8))
#       model.add(Dense(256, input_dim=4, activation='relu'))
#       model.add(Dropout(0.8))
#       model.add(Dense(512, input_dim=4, activation='relu'))
#       model.add(Dropout(0.8))
#       model.add(Dense(256, input_dim=4, activation='relu'))
#       model.add(Dropout(0.8))
#       model.add(Dense(128, input_dim=4, activation='relu'))
#       model.add(Dropout(0.8))
#       # model.add(Dense(64, activation='tanh')) # can be used…
#       model.add(Dense(2, activation='softmax'))
#       model.compile(loss='categorical_crossentropy',
#                      optimizer=Adam(lr=0.001), metrics=['accuracy'])


# print(model.input)
# model.fit(inputData, outputData, verbose=1, epochs=100)
# model.save('gamemodel.h5')
