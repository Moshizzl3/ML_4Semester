import os
import gym
from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import DummyVecEnv
from stable_baselines3.common.evaluation import evaluate_policy

envName = 'CartPole-v0'
episodes = 5

# Train model
# get log paths to display in tensorboard
log_path = os.path.join("training", "logs")
env = gym.make(envName)
env = DummyVecEnv([lambda: env])

PPO_Path = os.path.join('training', 'saved_models', 'PPO_Model_Carpole')

model = PPO.load(PPO_Path, env=env)

for episode in range(1, episodes + 1):
    observation = env.reset()
    done = False
    score = 0
    while not done:
        action, _ = model.predict(observation)
        observation, reward, done, info = env.step(action)
        score += reward
        env.render()
    print('episode:{} score: {}'.format(episode, score))
env.close()