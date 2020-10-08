import argparse
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import config

from torch.autograd import Variable

from enum import IntEnum
from scipy.signal import butter, lfilter, freqz

# In loss, use -train_target_signal for compensating signal,
# Use positive for making clone

# Define enum, so it is easy to update meaning of data indices
class Channel(IntEnum): # Channels in data block
    ANGLE = 0
    SIGNAL1 = 1 # torque / speed

# Can be from  to data length
future = 10 # How many steps to future (shift)


# Filter requirements.
order = 1
fs = 50.0       # sample rate, Hz
cutoff = 2.0  # desired cutoff frequency of the filter, Hz
def butter_lowpass(cutoff, fs, order=5):
    nyq = 0.5 * fs
    normal_cutoff = cutoff / nyq
    b, a = butter(order, normal_cutoff, btype='low', analog=False)
    return b, a

def butter_lowpass_filter(data, cutoff, fs, order=5):
    b, a = butter_lowpass(cutoff, fs, order=order)
    y = lfilter(b, a, data)
    return y

def moving_average(a, n=3) :
    ret = np.cumsum(a, dtype=float)
    ret[n:] = ret[n:] - ret[:-n]
    return ret[n - 1:] / n

# This model tries to learn a periodical signal from provided input data.
# After learning, the model can predict future values of similar signal.
# Generate_data.py can be used to generate input data for the model.

# number of recordings x rotations x signal length
# 
# Saved data: [angle, signal1]


# TODO: output 2x input, has input and the predictions after

# NOTE: Currently learns to predict value
# TODO: Rework loss. Has to compensate!
class Flatten(nn.Module):
    def forward(self, input):
        return input.view(input.size(0), -1)

class Sequence(nn.Module):
    def __init__(self, hidden=32):
        super(Sequence, self).__init__()
        self.hidden = hidden
        self.linear1 = nn.Linear(500, 50)
        self.linear2 = nn.Linear(3750, 999)
        self.conv1 = nn.Conv1d(2, hidden, 5, stride=2, padding=2)
        self.flatten = Flatten()
        print("Using %d hidden layers..." % hidden)

    # NN structure follows pytorch sequence example
    def forward(self, input, future = 0):
        x = F.relu(self.conv1(input))
        x = self.linear1(x)
        x = self.flatten(x)
        x = self.linear2(x) # [10, 999]
        output = x

        # if future > 0:
        # Could also do future forwarding + shifting here....
        return output

def plot2(data, filename):
    try:
        data = data.detach().numpy()
    except:
        pass # if already numpy data, don't detach()

    plt.figure(figsize=(15,7))
    x = np.arange(data.shape[1])
    y = data[0, :] # take first signal vec
    plt.plot(x, y, 'b', linewidth=2.0)
    plt.savefig(filename)
    plt.close()

# Plots input data
def plot3(data, filename):
    plt.figure(figsize=(15,7))
    x = np.arange(data.shape[2])
    y = data[0, 1, :] # take first signal vec
    plt.plot(x, y, 'b', linewidth=2.0)
    plt.savefig(filename)
    plt.close()


class Model(object):
    def __init__(self):
        self.seq = Sequence(75) #config.hidden_layers = N or repetitions
        self.seq.double()
        self.criterion = nn.MSELoss()
        # use LBFGS as optimizer since we can load the whole data to train
        self.optimizer = optim.LBFGS(self.seq.parameters(), lr=config.learning_rate)
        self.future = 1000

    # One step forward shift for signals
    # This should be tested...
    def shift(self, new_tensor, old_tensor):
        # Replace old input values with shifted signal data; old_tensor cannot be overwritten directly!
        tensor = old_tensor.clone() # keep graph
        tensor[:, Channel.SIGNAL1, :] = new_tensor[:, :] # shift signal (new values are in new_tensor, just assign). Currently mby wrong!!!
        tensor[:, Channel.ANGLE, :-1] = old_tensor[:, Channel.ANGLE, 1:] # shift angle. New angle values can be taken from old tensor with shift
        return tensor

    # In prediction, we propagate NN multiple times.
    def predict(self, test_input, test_target=None):
        if future > 0:
            with torch.no_grad(): # Do not update network when predicting
                for i in range(0, future-1):
                    print("shift:", i)
                    out = self.seq(test_input)
                    out = self.shift(out, test_input)
                out = self.seq(out) # Do no shigt on last forward
                if test_target is not None:
                    #print("[train] out:", out.size(), "target:", test_target.size())
                    shift = test_target.size(2) - future
                    test_target_signal = test_target[:, Channel.SIGNAL1, :shift]
                    loss = self.criterion(out[:, :shift], test_target_signal) # This makes no sense
                    print("prediction loss:", loss.item())
    
                out = self.shift(out, test_input) # Possibly extra forwad and should be removed, just to get dimensions right
                y = out.detach().numpy()
            print("y:", y.shape)
            return y #[:, 0] # return the 'new' prediction value
        else:
            return test_input

    def train(self, train_input, train_target):
        def closure():
            self.optimizer.zero_grad()
            out = self.seq(train_input)
            #print("[train] out:", out.size(), "target:", train_target.size())
            shift = train_target.size(2) - future
            train_target_signal = train_target[:, Channel.SIGNAL1, :shift]
            out = out[:, :shift]
            loss = self.criterion(out, train_target_signal) # flip: compensation loss
            #loss = self.criterion(out, train_target) # learn to predict loss
            plot2(out, "predictions/train_prediction.svg")

            print("loss:", loss.item())
            loss.backward()
            return loss
        self.optimizer.step(closure)

def plot(input_data, future, output, iteration):
    print("input_data shape:", input_data.shape)
    print("output_data shape:", output.shape)
    #print("tweaked output shape:", output[0, 0, -999:])

    input_length = input_data.size(2)
    plt.figure(figsize=(15,7))
    plt.xlabel(r"Angle ($\theta_m$)", fontsize=18, labelpad=5)
    plt.ylabel("Torque [pu.]", fontsize=18, labelpad=5)
    plt.xticks(fontsize=18)
    plt.yticks(fontsize=18)
    # def draw(yi, color):
    #     x = np.arange(input_length) #input_length
    #     #plt.plot(x, yi[:input_length], color, linewidth = 2.0)
    #     #plt.plot(np.arange(input_length, input_length + future), yi[input_length], color + ':', linewidth = 2.0)
    #draw(output[0, :future], 'b') # pick signal from first batch
    x1 = np.arange(input_length) #input_length

    x2 = np.arange(input_length, input_length + output.shape[1])
    plt.plot(x1, input_data[0, 1, :])
    plt.plot(x2, output[0, :])

    plt.savefig("predictions/prediction%d.svg" % iteration)
    plt.close()


# expects: ten batch 
def test(input, target):
    print("Input shape:", input.shape)
    print("Target shape:", target.shape)

    # if(input.shape == torch.Size(10, 2, 999)):
    #     print("Shape is correct!")


    # Check signal first value
    assert (target[0, 1, 0] == input[0, 1, 1]), "First item should be same"

    assert (target[9, 1, 0] == input[9, 1, 1]), "First item should be same"

    # Check signal last value
    assert (target[0, 1, -2] == input[0, 1, -1]), "Last item should be same"

    assert (target[9, 1, -2] == input[9, 1, -1]), "Last item should be same"



def main(args):
    # Load data.
    # input: x[i], target: x[i+1] in 1d-plane
    # One step lag allows guessing and learning.
    # Extract only signal stack from 'data block' for target
    train_data_filename = config.datafile + ".pt"
    test_data_filename = "testdata.pt"
    train_data = torch.load(train_data_filename)
    test_data = torch.load(test_data_filename)
    test_input = torch.from_numpy(test_data[..., :-1] )
    test_target = torch.from_numpy(test_data[..., 1:]) # match only signal values (ignore angle)
    train_input = torch.from_numpy(train_data[..., :-1])
    train_target = torch.from_numpy(train_data[..., 1:])

    test(train_input, train_target)
    test(test_input, test_target)

    print("train_input:", train_input.size())
    print("train target", train_target.size())
    

    # Create a new model
    model = Model()

    # Run
    for i in range(args.steps):
        print("STEP:", i)

        # 1) Let the model learn
        model.train(train_input, train_target)
        plot3(train_input, "predictions/train_input.svg")

        # 2) Check how model is performing
        y = model.predict(test_input, test_target)
        print("Target shape:", y[:, 1, :].shape)

        plot2(y[:, 1, :], "predictions/test_prediction.svg")
        plot3(test_input, "predictions/test_input.svg")


        # 3) Visualize performance
        plot(test_input, model.future, y[:, 1, :], i)
        #plot2(y, i)

    # We know that pulsation pattern should be relatively smooth.
    # Filter out the noise with low pass filter. This is better to do manually,
    # Since we already know what we want.
    angle = moving_average(y[0, 0, :], n=1)
    signal = moving_average(y[0, 1, :], n=1)
    #y = butter_lowpass_filter(y, cutoff, fs, order)

    plt.plot(np.arange(len(signal)), signal, color='green')
    plt.plot(np.arange(len(signal)), angle, color='blue')

    plt.savefig("predictions/filetered.svg")

    # Save result
    torch.save(model.seq.state_dict(), "predictions/compensator.mdl")
    np.savetxt("compensation-pattern.csv", np.array([angle, signal]), delimiter=",")

if __name__ == "__main__":
    
    #inputs = Variable(torch.rand(3, 5, 20)) # seq_len x batch_size x input_size
    parser = argparse.ArgumentParser()
    parser.add_argument("--steps", type=int, default=15, help="steps to run")
    args = parser.parse_args()
    main(args)
