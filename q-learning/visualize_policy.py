import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import gym
import envs

th_grid = np.linspace(0, 1, 100)
q_actions = np.linspace(-0.23, 0.23, 7)

# Indexing starts from 0
def getClosestIdx(list, value):
    return (np.abs(list - value)).argmin()

# Approx rotate the rotor to beginning.
def runToStart(sim):
    theta = sim.signal.getRotorElectricalAngleCtrl()
    while theta > 0.001:
        sim.command.step(0.0001)
        theta = sim.signal.getRotorElectricalAngleCtrl()
    return theta

def selectBestAction(qtable, angle):
    idx = np.argmax(qtable[angle])
    return idx

def oneRotation():
    Ilmarinen = gym.make('IlmarinenRawQlr-v2')
    sim = Ilmarinen.api
    qtable = np.load("qtable.npy")

    speed = 0.02
    sim.command.setSpeedReference(speed)
    sim.command.toggleQlr(True)

    # Start sim and let it to settle
    sim.command.step(30)
    theta = runToStart(sim)

    thetas = []
    actions = []
    angle_prev = None

    # Log data
    while theta < 0.999:
        theta = sim.signal.getRotorElectricalAngleCtrl()
        angle = getClosestIdx(th_grid, theta)
        if angle != angle_prev:
            angle_prev = angle
            thetas.append(theta)
            actions.append(selectBestAction(qtable, angle))
        sim.command.step(0.0001) # sampling interval
    del sim # free resources
    return thetas, actions

def anglePlot(X, Y):
    fig, ax = plt.subplots(figsize=(16,10), dpi=100)
    plt.grid(True, linestyle='dotted', color='silver')
    ax.plot(X, Y, '-o', label='Pulsating torque', linewidth=1.0, markersize=4, color='blue')
    plt.ylabel("Torque [pu.]", fontsize=18)
    plt.xlabel(r"Rotor angle ($\theta_{e}$)", labelpad=-5, fontsize=20)
    ax.xaxis.set_major_locator(plt.MaxNLocator(5))
    labels = [0, 0, r'$\frac{\pi}{2}$', r'$\pi$', r'$\frac{3\pi}{2}$', r'$2\pi$']
    ax.set_xticklabels(labels, fontsize=20)
    plt.yticks([-0.23, -0.115, 0, 0.115])  # Set label locations.
    plt.tick_params(labelsize=20)
    fig.tight_layout()

def main():
    angles, actions = oneRotation()
    print("Datapoits:", len(angles)) # should be equal to angle num

    actions2 = []
    for idx in actions:
        actions2.append(q_actions[idx])
    anglePlot(angles, actions2)
    plt.show(block=True)

if __name__ == '__main__':
    main()
