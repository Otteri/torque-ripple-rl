import argparse
import torch
import numpy as np
import matplotlib
import matplotlib.pyplot as plt
import config
from enum import IntEnum
import gym
import ilmarinen
from model import Model # SignalCovNet

DPI = 100

# This model tries to learn a periodical signal from provided input data.
# After learning, the model can predict future values of similar signal.
# Generate_data.py can be used to generate input data for the model.
#
# number of recordings x rotations x signal length
# Saved data: [angle, signal1]
# Data referes to the three dimensional block of data, wherease signal is just a 1d vector

parser = argparse.ArgumentParser()
parser.add_argument("--steps", type=int, default=15, help="steps to run")
parser.add_argument("--show_input", default=False, action="store_true", help="Visualize input training data")
parser.add_argument("--invert", default=False, action="store_true", help="Invert learning outcome")
parser.add_argument("--use_sim", action='store_true', default=None, help="Use simulator for data generation")
args = parser.parse_args()

if args.use_sim:
    from runsim import recordRotations

ave_n = 5

def configure():
    # Configure
    if not args.show_input:
        matplotlib.use("Agg")

# Define enum, so it is easy to update meaning of data indices
class Channel(IntEnum): # Channels in data block
    ANGLE = 0
    SIGNAL1 = 1 # torque / speed

def moving_average(a, n=3) :
    ret = np.cumsum(a, dtype=float)
    ret[n:] = ret[n:] - ret[:-n]
    return ret[n - 1:] / n


def plot(actual_torque, pulsating_torque, compensation_torque):

    plt.figure(figsize=(15,7))
    plt.xlabel(r"Angle ($\theta_m$)", fontsize=18, labelpad=5)
    plt.ylabel("Torque [pu.]", fontsize=18, labelpad=5)
    plt.xticks(fontsize=18)
    plt.yticks(fontsize=18)
    
    x = np.arange(actual_torque.size) #input_length
    plt.plot(x, compensation_torque, "-", color='g', label="compensation_torque")
    plt.plot(x, pulsating_torque, "-", color='b', label="pulsating_torque")
    plt.plot(x, actual_torque, "-", color='r', label="actual_torque")

    plt.legend()

    plt.savefig("predictions/eval.pdf")
    plt.close()

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
    plt.plot(Fv, getFFT(data[0]), label='Pulsating torque', linewidth=0.8, color='darkorange')
    plt.plot(Fv, getFFT(data[1]), label='Actual torque (filtered)', linewidth=0.8, color='red')
    plt.plot(Fv, getFFT(data[2]), label='Compensation torque', linewidth=0.8, color='yellowgreen')

    plt.title(title)
    plt.xlabel("Frequency [Hz]")
    plt.ylabel("Amplitude [p.u.]")
    plt.legend(loc=1)
    plt.grid(True)
    plt.savefig("predictions/eval_FFT.pdf")
    plt.show(block=False)



# Avg-filter input signal, since it can be quite noisy and we don't want to try learn white noise.
# Add padding by copying first few values in the beginning, so the data vector length does not change.
# input data: [B, 2, N]
def preprocessBatch(input_data, n=1):
    filtered_data = input_data.clone()
    for i in range(0, input_data.size(1)):
        padded_input_data = torch.cat((input_data[i, Channel.SIGNAL1, 0:(n-1)], input_data[i, Channel.SIGNAL1, :]))
        filtered_data[i, Channel.SIGNAL1, :] = moving_average(padded_input_data, n=n)
    return filtered_data

def find_nearest_idx(array, value):
    array = np.asarray(array)
    idx = (np.abs(array - value)).argmin()
    return idx

# Gather a data batch with N-rows
# batch_num x signals_num x signal_len
def main(args, compensate=True):
    use_speed = True
    data_length = 1000
    model = Model(compensate)
    model.seq.load_state_dict(torch.load("predictions/compensator.mdl"))
    model.seq.eval()

    Ilmarinen = gym.make('IlmarinenRawQlr-v1')
    sim = Ilmarinen.api
    sim.command.setSpeedReference(0.02)

    # Preallocate space
    time = np.zeros(data_length)
    data = np.zeros((5, data_length), dtype=float)
    measurement_data = np.empty((1, 2, data_length), 'float64')
    #torch.from_numpy(data[..., :-1]

    sim.command.setCompensationCurrentLimit(1.0)
    sim.command.toggleQlr(False)
    sim.command.toggleLearning(True)

    # Start sim and let it to settle
    randomization = np.random.uniform(1, 5)
    sim.command.step(20 + randomization)

    # One iteration to collect initial data block
    for i in range(data_length):
        time[i] = sim.status.getTimeStamp()
        data[0][i] = sim.signal.getCoggingTorque()
        data[1][i] = sim.signal.getSimActualTorque()
        data[2][i] = sim.signal.getCompensationTorque()
        data[3][i] = sim.signal.getRotorMechanicalAngle()
        data[4][i] = sim.signal.getMeasuredSpeed()
        sim.command.step(0.001) # sampling interval

    if use_speed:
        measurement_data[0, :, :] = np.vstack((data[3], data[4])) # [angle, signal1]
        measurement_data = torch.from_numpy(measurement_data[..., 1:]) # 1: shift

    else:
        measurement_data[0, :, :] = np.vstack((data[3], data[1])) # [angle, signal1]
        measurement_data = torch.from_numpy(measurement_data[..., 1:]) # 1: shift



    for i in range(data_length-1):
        time[i] = sim.status.getTimeStamp()
        data[0][i] = sim.signal.getCoggingTorque()
        data[1][i] = sim.signal.getSimActualTorque()
        data[2][i] = sim.signal.getCompensationTorque()
        data[3][i] = sim.signal.getRotorMechanicalAngle()
        data[4][i] = sim.signal.getMeasuredSpeed()

        sim.command.step(0.001) # sampling interval

        if compensate:
            # Get look-up table, with single rotation
            predictions = model.predict(measurement_data) # initially zero

            #predictions = model.predict(measurement_data) # initially zero
            #print("Angle: ", predictions[0, Channel.ANGLE, i])
            #idx = find_nearest_idx(predictions[0, Channel.ANGLE, :], data[3][i]) # find nearest idx using angle
            #print("idx:", idx)
            #prediction = float(predictions[0, Channel.SIGNAL1, i])
            #prediction = predictions[0, Channel.SIGNAL1, idx] # get signal value using found idx

            prediction = float(predictions[0, Channel.SIGNAL1, i])
            sim.signal.setCompensationTorque(prediction) # check if: compensation compensation -> doesn't compensate may viz correctly


    del sim # free resources
    return time, data

if __name__ == "__main__":
    time, data = main(args)
    plot(data[1], data[0], data[2])
    amplitudeSpectrum(time, data, "FFT")
