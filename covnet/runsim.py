import matplotlib.pyplot as plt
import numpy as np
import torch
import gym
import envs

def runSimulator():
    N = 1000 # amount of samples
    Ilmarinen = gym.make('IlmarinenRawQlr-v0')
    sim = Ilmarinen.api
    sim.command.setSpeedReference(0.02)

    # Preallocate space
    time = np.zeros(N)
    data = np.zeros((4, N), dtype=float)
    
    sim.command.setCompensationCurrentLimit(1.0)
    # Allows simulator to set compensate
    sim.command.toggleQlr(False)
    sim.command.toggleLearning(False)

    # Start sim and let it to settle
    sim.command.step(30)

    # Log data
    for i in range(N):
        time[i] = sim.status.getTimeStamp()
        data[0][i] = sim.signal.getCoggingTorque()
        data[1][i] = sim.signal.getSimActualTorque()
        data[2][i] = sim.signal.getCompensationTorque()
        data[3][i] = sim.signal.getRotorMechanicalAngle()

        sim.command.step(0.001) # sampling interval

    del sim # free resources

    return time, data

def collectData(rotations=1):
    _, data = runSimulator()
    recorded_data = np.vstack((data[3], data[1])) # [angle, signal1]
    return recorded_data
