from python_interface import Ilmarinen
import matplotlib.pyplot as plt
import numpy as np
import torch

DPI = 200 # plot resolution

def runSimulator(use_ilc=False):
    N = 1000 # amount of samples
    sim = Ilmarinen.SandboxApi()
    sim.command.setSpeedReference(0.03)

    # Preallocate space
    time = np.zeros(N)
    data = np.zeros((4, N), dtype=float)
    train_data = np.zeros((10, 2, N), dtype=float)
    
    # Start sim and let it to settle
    sim.command.step(90)

    for j in range(0, 10):
        # Log data
        for i in range(N):
            time[i] = sim.status.getTimeStamp()
            data[0][i] = sim.signal.getCoggingTorque()
            #data[1][i] = sim.signal.getSimActualTorqueFiltered()
            data[1][i] = sim.signal.getSimActualTorqueFiltered()
            data[2][i] = sim.signal.getCompensationTorque()
            data[3][i] = sim.signal.getRotorMechanicalAngle()
            sim.command.step(0.001) # sampling interval
    
        train_data[j, 0, :] = data[3]
        train_data[j, 1, :] = data[1]

    del sim # free resources

    return time, data, train_data

# Create typical line chart
def lineChart(time, data):
    plt.rc('grid', linestyle='dotted', color='silver')
    fig, ax1 = plt.subplots(1, figsize=(8,3), dpi=DPI)

    time = time - time[0] # make time start at zero; plot looks better
    
    ax1.plot(time, data[2], label='Compensation torque', linewidth=0.8, color='green')
    ax1.plot(time, data[0], label='Pulsating torque', linewidth=0.8, color='blue')
    ax1.plot(time, data[1], label='Total torque (filtered)', linewidth=0.8, color='red')
    ax1.plot(time, data[3], label='Mechanical rotor angle', linewidth=0.8, color='blue')
    ax1.tick_params(axis='x', which='both', bottom=False, top=False, labelbottom=False)
    ax1.margins(0, .05)
    ax1.grid(True)

    handles, labels = ax1.get_legend_handles_labels()
    fig.legend(handles, labels, loc = 'upper center', ncol=3)
    fig.text(0.5, 0.04, 'Time [s]', ha='center')
    fig.text(0.02, 0.5, 'Amplitude [pu.]', va='center', rotation='vertical')

    plt.show(block=False)


def main():
    # Obtain data
    time, data, train_data = runSimulator()

    # Plot the data
    lineChart(time, data)

    # Wait user to close the plots
    plt.show(block=True)

    torch.save(train_data, open("training_data.pt", 'wb'))
    print("Generation complete. Data shape:", train_data.shape)

if __name__ == '__main__':
    main()
