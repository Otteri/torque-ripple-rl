import matplotlib.pyplot as plt
import numpy as np
import gym
import envs

DPI = 100 # Plot resolution

def runSimulator(use_compensator=False):
    N = 1000 # amount of samples
    Ilmarinen = gym.make('IlmarinenRawQlr-v0')
    sim = Ilmarinen.api
    sim.command.setSpeedReference(0.02)

    # Preallocate space
    time = np.zeros(N)
    data = np.zeros((20, N), dtype=float)

    if use_compensator:
        sim.command.toggleLearning(True)
        sim.command.toggleQlr(True)
        sim.command.step(999) # let learn
    else:
        sim.command.step(10) # let to settle

    # Log data
    for i in range(N):
        time[i] = sim.status.getTimeStamp()
        data[0][i] = sim.signal.getCoggingTorque()
        data[1][i] = sim.signal.getSimActualTorqueFiltered()
        data[2][i] = sim.signal.getCompensationTorque()
        sim.command.step(0.001) # sampling interval

    del sim # free resources

    return time, data

def plotData(time, data):
    window_name = "Ilmarinen run data"
    lineChart(window_name, time, data)
    amplitudeSpectrum(window_name, time, data)
    plt.show() # Block

# Create typical line chart
def lineChart(time, data, title):
    plt.rc('grid', linestyle='dotted', color='silver')
    plt.figure(figsize=(6.4,5.0), dpi=DPI)
    plt.plot(time, data[2], label='Compensation torque', linewidth=0.8, color='yellowgreen')
    plt.plot(time, data[0], label='Pulsating torque', linewidth=0.8, color='darkorange')
    plt.plot(time, data[1], label='Actual torque (filtered)', linewidth=0.8, color='red')

    plt.title(title)
    plt.xlabel("Time [s]")
    plt.ylabel("Amplitude [pu.]")
    plt.legend(loc=1)
    plt.grid(True)
    plt.show(block=False)

# Compute one sided FFT and plot the result
def amplitudeSpectrum(times, data, title):
    # First calculate L and Fv, then call this
    # function in order to obtain the FFT result
    def getFFT(signal):
        Y = np.fft.fft(signal)/L              # Transform
        P = np.abs(Y[0:len(Fv)])              # get magnitudes
        P[2:-1] = 2*P[2:-1]                   # double, due to one-sided spectrum
        return P
    
    plt.rc('grid', linestyle='dotted', color='silver')
    plt.figure(figsize=(6.4,5.0), dpi=DPI)
    T = times[1] - times[0]                   # sampling interval 
    L = times.size                            # data vector length
    Fn = (1/T)/2                              # calculate frequency
    Fv = np.linspace(0, 1, int(np.fix(L / 2)))*Fn  # frequency vector (one-sided)

    # Plot
    plt.plot(Fv, getFFT(data[2]), label='Compensation torque', linewidth=0.8, color='yellowgreen')
    plt.plot(Fv, getFFT(data[0]), label='Pulsating torque', linewidth=0.8, color='darkorange')
    plt.plot(Fv, getFFT(data[1]), label='Actual torque (filtered)', linewidth=0.8, color='red')

    plt.title(title)
    plt.xlabel("Frequency [Hz]")
    plt.ylabel("Amplitude [p.u.]")
    plt.legend(loc=1)
    plt.grid(True)
    plt.show(block=False)

def main():
    # Obtain data
    time1, data1 = runSimulator()
    time2, data2 = runSimulator(use_compensator=True)

    # Plot the data
    lineChart(time1, data1, "Torque pulsations - compensator disabled")
    amplitudeSpectrum(time1, data1, "Torque pulsations - compensator disabled")
    lineChart(time2, data2, "Torque pulsations - compensator enabled")
    amplitudeSpectrum(time2, data2, "Torque pulsations - compensator enabled")
    
    # Wait until user closes plots
    plt.show(block=True)

if __name__ == '__main__':
    main()
