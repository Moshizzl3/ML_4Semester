class Agent():
    def __init__(self, env):
        self.action_size = env.action_space.n
        print("action size", self.action_size)
    
    def getAction(self, state):
        # action = random.choice(range(self.action_size)) random choice action
        pole_angle = state[2]
        action = 0 if pole_angle < 0 else 1 # action where agent trys and correct depending on angle
        return action