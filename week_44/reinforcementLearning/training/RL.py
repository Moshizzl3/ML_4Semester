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
model = PPO('MlpPolicy', env, verbose=1, tensorboard_log=log_path)
model.learn(total_timesteps=20000)

PPO_Path = os.path.join('training', 'saved_models', 'PPO_Model_Carpole')
model.save(PPO_Path)

#Eval model

print(evaluate_policy(model, env, n_eval_episodes=10, render=True))
