import os
import gym
from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import DummyVecEnv
from stable_baselines3.common.evaluation import evaluate_policy

envName = 'CartPole-v0'
env = gym.make(envName)
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
del model
model = PPO.load(PPO_Path, env=env)

#Eval model

evaluate_policy(model, env, n_eval_episodes=10, render=True)


# for episode in range(1, episodes + 1):
#     state = env.reset()
#     done = False
#     score = 0

#     while not done:
#         env.render()
#         action = env.action_space.sample()
#         nState, reward, done, info = env.step(action)
#         score += reward
#     print('episode:{} score: {}'.format(episode, score))
# env.close()