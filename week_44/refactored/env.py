import gym
import random
import numpy as np
import tensorflow as tf
from keras.models import load_model
from statistics import mean, median
import time
import datetime
from keras.models import Sequential
from keras.layers import Dense, Dropout, BatchNormalization
from keras.models import load_model
import numpy as np
from keras.optimizers import Adam
from pathlib import Path

timeStep = 500

myEnv_name = "Acrobot-v1"  # env i want to use

env = gym.make(myEnv_name)  # create env

scores = []
numberOfGames = 250
score_requirement = -200  # This will be incremented stepwise, to make the model stronger
count = 0


class Agent():
    def __init__(self, env):
        self.is_discrete = \
            type(env.action_space) == gym.spaces.discrete.Discrete
        self.score = 0
        if self.is_discrete:
            self.action_size = env.action_space.n
            print("action size", self.action_size)
        else:
            self.action_low = env.action_space.low
            self.action_high = env.action_space.high
            self.action_shape = env.action_space.shape

    def getAction(self, state, model):
        if self.is_discrete:
            try:

                # first action is random, the rest from the model.
                # gets prediction from model, skal bruger næste gang.
                prediction = model.predict(np.array([state])).tolist()
                # print(prediction) # [[0.48285746574401855, 0.5171425938606262]] this is how the prediction looks like
                # max() gets the largest value, index() gets its index.
                indexOfGuess = prediction[0].index(max(prediction[0]))
                if (indexOfGuess == 0):
                    action = 0
                    output = [1, 0, 0]
                elif (indexOfGuess == 1):
                    action = 1
                    output = [0, 1, 0]
                elif (indexOfGuess == 2):
                    action = 2
                    output = [0, 0, 1]
            except:
                action = 0 if state[2] < 0 else 1
                if (indexOfGuess == 0):
                    action = 0
                    output = [1, 0, 0]
                elif (indexOfGuess == 1):
                    action = 1
                    output = [0, 1, 0]
                elif (indexOfGuess == 2):
                    action = 2
                    output = [0, 0, 1]

        else:
            action = np.random.uniform(
                self.action_low, self.action_high, self.action_shape)  # random choice action
        print(self.score)
        return [action, output]

    def trainAgent(self, newData):
        newList = np.array(newData)

        try:
            trainingData = np.load(
                'saved.npy', allow_pickle=True)
            inputData = trainingData[:, 0].tolist()
            outputData = trainingData[:, 1].tolist()
            forget = int(1*len(trainingData))
            concList = np.concatenate((trainingData, newList), axis=0)
            np.save('saved.npy', concList)
        except:
            inputData = newList[forget:, 0].tolist()
            outputData = newList[forget:, 1].tolist()
            np.save('saved.npy', newList)

        model = Sequential()  # creates new, empty model
        model.add(Dense(64, activation='relu'))
        model.add(Dense(128, activation='relu'))
        model.add(Dense(64, activation='relu'))
        # model.add(Dense(64, activation='tanh')) # can be used…1
        model.add(Dense(3, activation='linear'))
        model.compile(loss='mae',
                      optimizer=Adam(lr=0.0001), metrics=['accuracy'])
        print('New model created, no pre-trained model found...')

        model.fit(inputData, outputData, verbose=1, epochs=10)
        model.save('gamemodel.h5')

    # NOTE: if there is nothing to save, the saved file will be destroyed.


agent = Agent(env)
state = env.reset()  # return the initial state of the game
action = random.randint(0,3)
for x in range(numberOfGames):
    model = load_model('gamemodel.h5')
    print("-------------- number:", x)
    training_data = []
    agent.score = 0
    env.reset()
    game_memory = []
    for _ in range(timeStep):
        state, reward, done, info = env.step(action)
        action = agent.getAction(state, model)[0]  # get action from agent
        agent.score += reward
        env.render()
        # return tuple with diff info of game and sets next state
        if done:
            scores.append(agent.score)
            env.close()
            break
        game_memory.append([state, agent.getAction(state, model)[1]])
        # print (game_memory)
    if agent.score >= score_requirement:  # If a game does well, it is saved.
        for data in game_memory:  # Takes all data from game_memory and places it in training_data
            # This list will be saved to file.
            training_data.append([data[0].tolist(), data[1]])
        agent.trainAgent(training_data)
        score_requirement = 30 if score_requirement >50 else score_requirement + 2
        print('Agent score:', agent.score)
        print('score requirement:', score_requirement)
