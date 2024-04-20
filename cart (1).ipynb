import random
import gym
import numpy as np

from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Flatten
from tensorflow.keras.optimizers.legacy import Adam

from rl.agents import DQNAgent
from rl.policy import BoltzmannQPolicy
from rl.memory import SequentialMemory


# makes a environment of cartPole
env = gym.make("CartPole-v1")   

states = env.observation_space.shape[0]
#number of actions is 2 but we are dynamically pulling it
actions = env.action_space.n

#underlying machine learning model
model = Sequential()
model.add(Flatten(input_shape=(1,states)))
model.add(Dense(24, activation="relu"))
model.add(Dense(24, activation="relu"))
model.add(Dense(actions, activation="linear"))

#this agent is using the above model
agent = DQNAgent(
    model = model,
    memory = SequentialMemory(limit=50000, window_length=1),
    policy=BoltzmannQPolicy(),
    nb_actions=actions,
    nb_steps_warmup=10,
    target_model_update=0.01
)
#compiling using adam optimizer
agent.compile(Adam(learning_rate=0.1), metrics=["mae"])
agent.fit(env, nb_steps=50000, visualize=False, verbose=1)

res = agent.test(env,nb_episodes=10, visualize=True)
print(np.mean(res.history["episode_reward"]))
env.close()

"""
# through randomization we are going to try to solve the cartpole problem
# essentially by adding left or right force at random
episodes = 10
for episode in range(1, episodes+1):
    state = env.reset()
    done = False
    score = 0

    # while the cartpole is not in a done state
    # can do while True to see the complete fail 
    while not done:
        action = random.choice([0,1])
        # reward is the positive when the action we took was right
        # negative when its wrong and done shows the current state of the pole
        _, reward, done, _ =  env.step(action)
        score += reward
        env.render()
    print(f"Episode {episode}, Score: {score}")

env.close()
"""
