import matplotlib.pyplot as plt
import numpy as np
import torch
import gym
import ilmarinen

def runSimulator(data_length=1000, use_speed=False):
    Ilmarinen = gym.make('IlmarinenRawQlr-v2')
    sim = Ilmarinen.api
    sim.command.setSpeedReference(0.02)

    # Preallocate space
    time = np.zeros(data_length)
    data = np.zeros((5, data_length), dtype=float)
    
    sim.command.setCompensationCurrentLimit(1.0)
    sim.command.toggleQlr(False)
    sim.command.toggleLearning(False)

    # Start sim and let it to settle
    sim.command.step(30)

    # Log data
    for i in range(data_length):
        time[i] = sim.status.getTimeStamp()
        data[0][i] = sim.signal.getCoggingTorque()
        data[1][i] = sim.signal.getSimActualTorque()
        data[2][i] = sim.signal.getCompensationTorque()
        data[3][i] = sim.signal.getRotorMechanicalAngle()
        data[4][i] = sim.signal.getMeasuredSpeed()
        sim.command.step(0.001) # sampling interval
    
    if use_speed:
        recorded_data = np.vstack((data[4], data[1])) # [angle, signal1]
    else:
        recorded_data = np.vstack((data[3], data[1])) # [angle, signal1]

    del sim # free resources
    return recorded_data

def recordRotations(rotations=1, data_length=1000, use_speed=False):
    data = np.empty((rotations, 2, data_length), 'float64')
    for i in range(0, rotations):
        print("Collecting sample: {}/{}".format((i+1), rotations))
        data[i, :, :] = runSimulator(data_length, use_speed=use_speed)
    return data
