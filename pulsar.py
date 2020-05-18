import sys
import gym
import envs
import torch
import random
import argparse
import numpy as np
import matplotlib.pyplot as plt

# Hyperparameters
gamma = 0.99
alpha = 0.6
e = 500 # epsilon converges

# Grid parameters
angle_dt = 100 #1334 # Number of possible states with 2.5us step and 60rpm.
actions = np.linspace(-0.23, 0.23, 5) #7
th_min, th_max = 0, 1
th_grid = np.linspace(th_min, th_max, angle_dt)

def createGrid():
    print("Created a new Q-grid...")
    initial_q = 0.0
    qtable = np.zeros((angle_dt, len(actions))) + initial_q
    return qtable

def loadGrid(location):
    print("Loading model from '{}'".format(location))  
    qtable = np.load(location)
    print("Grid signature:", getGridSignature(qtable))
    return qtable

# Indexing starts from 0
def getClosestIdx(list, value):
    return (np.abs(list - value)).argmin()

# Calculate 'signature' by summing all weights. Useful for checking
# if load succeed. Simple but effective way, which is also easy to
# implement on C++ side. Use only int due to floating point deviation.  
def getGridSignature(grid):
    total = 0.0
    for value in np.nditer(grid): 
        total += value
    return int(total)

# Parse script arguments
def parseArgs(args=sys.argv[1:]):
    parser = argparse.ArgumentParser()
    parser.add_argument("--test", "-t", type=str, default=None, help="Model to be tested")
    parser.add_argument("--episodes", "-ep", type=int, default=500, help="Number of episodes to train for")
    parser.add_argument("--render", action='store_true', default=None, help="Render compensation waveform")
    parser.add_argument("--load", type=str, default=None, help="Load model and continue training")
    parser.add_argument("--no_compensation", action='store_true', default=None, help="Does not use or train compensator")
    return parser.parse_args(args)

# Note: cannot plot if rendering
def plotReward(reward_history, average_reward_history):
    plt.close('all')
    episodes = np.arange(len(reward_history))
    fig = plt.figure()
    plt.plot(episodes, reward_history, label='Instantaneous reward', color='lightblue', alpha=0.8)
    plt.plot(episodes, average_reward_history, label='Reward average' ,linewidth=0.8, color='b')
    plt.xlabel('Episode number')
    plt.ylabel('Reward')
    plt.legend(loc='lower right')
    fig.savefig("reward-history.svg")

# Returns best action if random number [0, 1] is larger than epsilon.
# Hence, to always get best action, give epsilon < 0.
def selectAction(qtable, angle, epsilon):
    if random.random() > epsilon:
        return np.argmax(qtable[angle])
    else:
        return random.randrange(len(actions))

def test(env, qtable, episodes, render=False):
    print("Testing...")
    for episode_number in range(1, episodes):
        reward_sum, timesteps = 0, 0
        done = False
        state = env.reset()
        while not done:
            angle = getClosestIdx(th_grid, state[1])
            action_idx = selectAction(qtable, angle, -1)
            new_state, reward, done, info = env.step(actions[action_idx])
            
            state = new_state
            reward_sum += reward
            timesteps += 1

            # Collect data for rendering
            if render and episode_number % 1 == 0:
                env.render()
    
        print("Episode {}, total reward {:.2f}, trip {}, torque avg. {}".format(episode_number, reward_sum, info, 0))

def train(env, qtable, episodes, render=False):
    reward_history = []
    average_reward_history = []
    max_rewrd = -10 # some initial guess which skips early values

    for episode_number in range(1, episodes+1):
        reward_sum, timesteps = 0, 0
        done = False
        state = env.reset()
        while not done:
            # Decide action
            epsilon = e / (e + episode_number)
            angle = getClosestIdx(th_grid, state[1])
            action_idx = selectAction(qtable, angle, epsilon)

            # Step forward
            new_state, reward, done, info = env.step(actions[action_idx])
            new_angle = getClosestIdx(th_grid, new_state[1])

            # Update grid
            qtable[angle][action_idx] += alpha * (reward + gamma * np.max(qtable[new_angle]) - 
            qtable[angle][action_idx])
                  
            state = new_state
            reward_sum += reward
            timesteps += 1

            # Collect data for rendering
            if render and episode_number % 1 == 0:
                env.render()

        if episode_number % 10 == 0 and episode_number > 50:
            print("Episode {}, total reward {:.2f}, trip {}, timesteps {}, torque avg. {}".format(episode_number, reward_sum, info, timesteps, 0))


        if reward_sum < -1.0:
            reward_history.append(reward_sum)
            avg = np.mean(reward_history[-100:]) if episode_number > 100 else np.mean(reward_history)
            average_reward_history.append(avg)

        if episode_number % 50 == 0:
        #if reward_sum >= max_rewrd and info > 99:
        #    max_rewrd = reward_sum
            np.save("qtable.npy", qtable)
            print("Grid signature:", getGridSignature(qtable))
            print("Epsilon:", epsilon)
            if not render: # cannot plot & render simultaneously
                plotReward(reward_history, average_reward_history)

def main(args):

    # Select correct environment
    if args.no_compensation and args.render:
        env = gym.make('IlmarinenEnv3-v0')
    elif args.render:
        env = gym.make('IlmarinenEnv-v0')
    else:
        env = gym.make('IlmarinenEnv2-v0')

    # Load or create a new Qtable?
    if args.test is None:
        qtable = loadGrid(args.load) if args.load is not None else createGrid()
        train(env, qtable, args.episodes, args.render)
    else:
        qtable = loadGrid(args.test)
        test(env, qtable, args.episodes, args.render)

if __name__ == "__main__":
    args = parseArgs()
    main(args)
