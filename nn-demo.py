from python_interface import Ilmarinen
import matplotlib.pyplot as plt
import numpy as np
import torch
from numpy import genfromtxt

import config
from datagenerator import plotRecordedData

PI2 = 2*np.pi
DPI = 200 # plot resolution
SHIFT = 0 #25

def find_nearest_idx(array, value):
    array = np.asarray(array)
    idx = (np.abs(array - value)).argmin()
    return idx


def getCompensationValue(array, rotor_angle):
    angle_array = array[0, :]
    idx = find_nearest_idx(angle_array, rotor_angle)
    #print("idx:", idx) # should increase smoothly

    if idx + SHIFT < len(array[0, :]):
        idx = idx + SHIFT # shift
    else:
        idx = idx - len(array[0, :])

    return array[1, idx]

def runSimulator(compensation_pattern):
    N = 1000 # amount of samples
    sim = Ilmarinen.SandboxApi()
    sim.command.setSpeedReference(0.02)

    # Preallocate space
    time = np.zeros(N)
    data = np.zeros((4, N), dtype=float)
    
    sim.command.setCompensationCurrentLimit(1.0)
    # Allows simulator to set compensate
    sim.command.toggleQlr(True)
    sim.command.toggleLearning(True)

    # Start sim and let it to settle
    sim.command.step(90)

    # Log data
    for i in range(N):
        time[i] = sim.status.getTimeStamp()
        data[0][i] = sim.signal.getCoggingTorque()
        data[1][i] = sim.signal.getSimActualTorque()
        data[2][i] = sim.signal.getCompensationTorque()
        data[3][i] = sim.signal.getRotorMechanicalAngle()

        sim.command.step(0.001) # sampling interval

        best_value = getCompensationValue(compensation_pattern, data[3][i]) # Now compensation comes one step too late
        #print("best value:", best_value)
        sim.signal.setAction(best_value)

    del sim # free resources

    return time, data

def getWindowName(name=None):
    default ="Simulation - Ilmarinen"
    name = name if name else default
    return name

# Create typical line chart
def lineChart(time, data):
    plt.rc('grid', linestyle='dotted', color='silver')
    fig, ax1 = plt.subplots(1, figsize=(8,3), dpi=DPI)

    time = time - time[0] # make time start at zero; plot looks better
    
    ax1.plot(time, data[2], label='Compensation torque', linewidth=0.8, color='green')
    ax1.plot(time, data[0], label='Pulsating torque', linewidth=0.8, color='blue')
    ax1.plot(time, data[1], label='Actual torque', linewidth=0.8, color='red')
    ax1.plot(time, data[3], label='Mechanical rotor angle', linewidth=0.8, color='magenta')
    ax1.tick_params(axis='x', which='both', bottom=False, top=False, labelbottom=False)
    ax1.margins(0, .05)
    ax1.grid(True)

    handles, labels = ax1.get_legend_handles_labels()
    fig.legend(handles, labels, loc = 'upper center', ncol=4, prop={'size': 8})
    fig.text(0.5, 0.04, 'Time [s]', ha='center')
    fig.text(0.02, 0.5, 'Amplitude [pu.]', va='center', rotation='vertical')

    plt.show(block=True)

def moving_average(a, n=3) :
    ret = np.cumsum(a, dtype=float)
    ret[n:] = ret[n:] - ret[:-n]
    return ret[n - 1:] / n

def runPulsationModel(compensation_pattern):
    def sample(angle):
        harmonics = config.harmonics
        torque = 0.0
        for harmonic_n in harmonics:
            magnitue = harmonics[harmonic_n]
            torque += magnitue * np.cos(harmonic_n * angle)
        return torque

    rotations = 1
    L = config.L
    N = config.N
    step_size = config.step_size
    signal_data = np.empty(L, 'float64')
    rotation_data = np.empty(L, 'float64')
    requested_sample_amount = L
    current_sample_num = 0
    compensation = 0
    compensation_prev = 0
    compensation_prev_prev = 0
    compensate = True

    time = np.zeros(L)
    data = np.zeros((4, L), dtype=float)

    # Do one approximately full mechanical rotation
    # Might not be precisely full rotation, due to added noise
    rotor_angle = np.random.uniform(0, PI2) # Start from random angle
    while(current_sample_num < requested_sample_amount):
        # moving average filter, seconod order

        if compensate:
            compensation = (1.0/3) * (getCompensationValue(compensation_pattern, rotor_angle) + compensation_prev + compensation_prev_prev)
            compensation_prev = compensation
            compensation_prev_prev = compensation_prev

        rotation_data[current_sample_num] = rotor_angle
        signal_data[current_sample_num] = sample(rotor_angle) + compensation

        time[current_sample_num] = time[current_sample_num -1] + 0.001
        data[3][current_sample_num] = rotor_angle
        data[0][current_sample_num] = sample(rotor_angle)
        data[2][current_sample_num] = compensation
        data[1][current_sample_num] =  sample(rotor_angle) + compensation # actual torque

        noise = np.random.uniform(-config.noise, config.noise) # simulate encoder noise (%)
        rotor_angle += step_size + noise
        current_sample_num += 1

        # Make angle to stay within limits: [0, PI2]
        if (rotor_angle >= PI2):
            rotor_angle = 0

    return time, data


def main():
    use_sim = False

    # Obtain data
    compensation_pattern = genfromtxt('predictions/compensation-pattern.csv', delimiter=',')

    fig, ax1 = plt.subplots(1, figsize=(8,3), dpi=DPI)
    x = np.arange(compensation_pattern.shape[1])    
    ax1.plot(x, compensation_pattern[0, :], label='Rotor angle', linewidth=0.8, color='green')
    ax1.plot(x, compensation_pattern[1, :], label='Compensation pattern', linewidth=0.8, color='blue')
    plt.show(block=True)

    if use_sim:
        time, data = runSimulator(compensation_pattern)
    else:
        #data = np.empty((1, 2, config.L), 'float64')
        time, data = runPulsationModel(compensation_pattern)

    lineChart(time, data)

    # Wait user to close the plot
    plt.show(block=True)

if __name__ == '__main__':
    main()
