import gym
from keras.optimizers import Adam
from kerasRL import buildModel, buildAgent




env = gym.make("MountainCar-v0")
states = env.observation_space.shape[0]
actions = env.action_space.n

model = buildModel(states, actions)
agent = buildAgent(model, actions)
agent.compile(Adam(lr=1e-3), metrics=['mae'])

agent.load_weights('./agentWeights.h5f')

scores = agent.test(env, nb_episodes=5, visualize=True)

