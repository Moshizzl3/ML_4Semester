import gym
import random
import numpy as np
from statistics import mean, median

# load this particular game. Other games are available. (render_mode="rgb_array")
env = gym.make('CartPole-v0')
state = env.reset()
goal_steps = 200  # Timesteps a game runs
# Timesteps a game needs to be accepted in to training set. 50 is good
score_requirement = 50
initial_games = 10000  # 1000 is good
numberOfData = 0  # Only for examining number of data


def intial_population():
    training_data = []
    scores = []
    # A list to hold scores for those games, which made it. So we can print them later.
    accepted_scores = []
    for _ in range(initial_games):
        score = 0
        game_memory = []
        prev_observation = []
        observation = env.reset()  # reset game
        for _ in range(goal_steps):
            # ---------------- will show game on screen --------------------------------
            env.render()
            # ---------------- will show game on screen --------------------------------
            # will start with random number from 0 to 1. 0 = push car left. 1 = push car right.
            action = 0 if observation[2] < 0 else 1
            observation, reward, done, info = env.step(
                action)  # makes the game move one Timestep
            # the observation object contains these four data points (source):

            if len(prev_observation) > 0:
                # only add to game memory, if previous observation exists.
                game_memory.append([prev_observation, action])

            prev_observation = observation
            score += reward
            if done:  # The pole has fallen too far to either side. Game is stopped.
                break

        if score >= score_requirement:  # If a game was successful, it will be saved
            accepted_scores.append(score)  # list to print the scores
            for data in game_memory:  # Takes all data from game_memory and places it in training_data
                if data[1] == 1:
                    output = [0, 1]  # transforms data to One-Hot encoding.
                elif data[1] == 0:
                    output = [1, 0]  # transforms data to One-Hot encoding.
                # This list is saved to file.
                training_data.append([data[0].tolist(), output])
        env.reset()  # Reset game
        # saves the score, even though it did not make it above score_requirement.
        scores.append(score)

    # saves training_data to file.
    np.save('saved.npy', np.array(training_data))
    print('-----------------------------------------------')
    print('Number of accepted score: ' + str(len(accepted_scores)))
    print('Number of training_data: ' + str(len(training_data)))
    print('Average accepted score: ' + str(mean(accepted_scores)))
    print('Average  score: ' + str(mean(scores)))
    print('Median  score: ' + str(median(scores)))
    print('Max  score: ' + str(max(scores)))
    print('Min  score: ' + str(min(scores)))
    print('-----------------------------------------------')
    return training_data


train = intial_population()

# see what is saved in training_data
trainingData = np.load('saved.npy', allow_pickle=True)
inputData = trainingData[:, 0].tolist()
outputData = trainingData[:, 1].tolist()
print('-----------------------------------------------')
print('See one input data: ' + str(inputData[0]))
print('See one output data: ' + str(outputData[0]))
print('-----------------------------------------------')
