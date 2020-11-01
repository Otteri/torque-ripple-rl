import matplotlib.pyplot as plt
import numpy as np
import gym
import envs

DPI = 200 # plot resolution

def runSimulator(use_ilc=False):
    Ilmarinen = gym.make('IlmarinenRawILC-v0')
    sim = Ilmarinen.api

    N = 1000 # amount of samples
    sim.command.setSpeedReference(0.03)

    # Preallocate space
    time = np.zeros(N)
    data = np.zeros((20, N), dtype=float)
    
    # Use ILC?
    if use_ilc:
        sim.command.toggleILC(True)
        sim.command.setFiiILC(10.0)
        sim.command.setGammaILC(5.0)
        sim.command.setCompensationCurrentLimit(0.5)

    # Start sim and let it to settle
    sim.command.step(90)

    # Log data
    for i in range(N):
        time[i] = sim.status.getTimeStamp()
        data[0][i] = sim.signal.getCoggingTorque()
        data[1][i] = sim.signal.getSimActualTorqueFiltered()
        data[2][i] = sim.signal.getCompensationTorque()
        sim.command.step(0.002) # sampling interval

    del sim # free resources

    return time, data

def getWindowName(name=None):
    default ="Simulation - Ilmarinen"
    name = name if name else default
    return name

# Create typical line chart
def lineChart2(time1, data1, time2, data2, name=None):
    plt.rc('grid', linestyle='dotted', color='silver')
    fig, (ax1, ax2) = plt.subplots(2, figsize=(6.4,5.0), dpi=DPI)
    fig.canvas.set_window_title(getWindowName(name)) 

    time1 = time1 - time1[0] # make time start at zero; plot looks better
    time2 = time2 - time2[0]
    
    ax1.plot(time1, data1[2], label='Compensation torque', linewidth=0.8, color='green')
    ax1.plot(time1, data1[0], label='Pulsating torque', linewidth=0.8, color='blue')
    ax1.plot(time1, data1[1], label='Actual torque', linewidth=0.8, color='red')
    ax1.tick_params(axis='x', which='both', bottom=False, top=False, labelbottom=False)
    ax1.margins(0, .05)
    ax1.grid(True)

    ax2.plot(time2, data2[2], label='2Compensation torque', linewidth=0.8, color='green')
    ax2.plot(time2, data2[0], label='2Pulsating torque', linewidth=0.8, color='blue')
    ax2.plot(time2, data2[1], label='2Actual torque', linewidth=0.8, color='red')
    ax2.tick_params(axis='x', which='both', bottom=True, top=False, labelbottom=True, labeltop=False)
    ax2.margins(0, .05)
    ax2.grid(True)

    handles, labels = ax1.get_legend_handles_labels()
    fig.legend(handles, labels, loc = 'upper center', ncol=3)
    fig.text(0.5, 0.04, 'Time [s]', ha='center')
    fig.text(0.02, 0.5, 'Amplitude [pu.]', va='center', rotation='vertical')

    plt.show(block=False)
    plt.savefig("picture1.svg")

def main():
    # Obtain data
    time1, data1 = runSimulator()
    time2, data2 = runSimulator(use_ilc=True)

    # Plot the data
    lineChart2(time1, data1, time2, data2)

    # Wait user to close the plots
    plt.show(block=True)

if __name__ == '__main__':
    main()
