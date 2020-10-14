from python_interface import Ilmarinen
import matplotlib.pyplot as plt
import numpy as np
import torch
from numpy import genfromtxt

DPI = 200 # plot resolution
SHIFT = 25

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

def runSimulator():
    N = 1000 # amount of samples
    sim = Ilmarinen.SandboxApi()
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

        # Compensate:
        #best_value = getCompensationValue(compensation_pattern, data[3][i]) # Now compensation comes one step too late
        #print("best value:", best_value)
        #sim.signal.setAction(best_value)

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
    ax1.plot(time, data[3], label='Mechanical rotor angle', linewidth=0.8, color='blue')
    ax1.tick_params(axis='x', which='both', bottom=False, top=False, labelbottom=False)
    ax1.margins(0, .05)
    ax1.grid(True)

    handles, labels = ax1.get_legend_handles_labels()
    fig.legend(handles, labels, loc = 'upper center', ncol=3)
    fig.text(0.5, 0.04, 'Time [s]', ha='center')
    fig.text(0.02, 0.5, 'Amplitude [pu.]', va='center', rotation='vertical')

    plt.show(block=True)


def collectData(rotations=1):
    time, data = runSimulator()
    recorded_data = np.vstack((data[3], data[1])) # [angle, signal1]
    return recorded_data


    # Plot the data
    #lineChart(time, data)

    # Wait user to close the plots
    #plt.show(block=True)
