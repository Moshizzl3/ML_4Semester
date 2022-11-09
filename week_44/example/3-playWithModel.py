from keras.models import load_model
import numpy as np
import gym
import random
from statistics import mean, median
import time


model = load_model('gamemodel.h5')

env = gym.make('Acrobot-v1')

env.reset()
goal_steps = 500
env._max_episode_steps = 500  # Default is 200


action = random.randrange(0, 3)  # The first action is random.
scores = []
score = 0
training_data = []
numberOfGames = 10
score_requirement = -500  # This will be incremented stepwise, to make the model stronger
count = 0

for x in range(numberOfGames):
    env.reset()
    score = 0
    game_memory = []

    for t in range(goal_steps):
        # first action is random, the rest from the model.
        observation, reward, done, info = env.step(action)
        env.render()  # displays game on screen
        # gets prediction from model, skal bruger næste gang.
        prediction = model.predict(np.array([observation])).tolist()
        # print(prediction) # [[0.48285746574401855, 0.5171425938606262]] this is how the prediction looks like
        # max() gets the largest value, index() gets its index.
        indexOfGuess = prediction[0].index(max(prediction[0]))
        score += reward
        if (indexOfGuess == 0):
            action = 0
            output = [1, 0, 0]
        elif (indexOfGuess == 1):
            action = 1
            output = [0, 1, 0]
        elif (indexOfGuess == 2):
            action = 1
            output = [0, 0, 1]
        if done:
            scores.append(score)
            env.close()
            break
        game_memory.append([observation, output])
        count = count + 1
        print("Time: ", t)

print('Average score', mean(scores))
print('Median score', median(scores))
print('Min score', min(scores))
print('Max score', max(scores))
print('Number of training_data: ' + str(len(training_data)))
