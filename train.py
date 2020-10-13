import argparse
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.autograd import Variable
import numpy as np
import matplotlib
import matplotlib.pyplot as plt
import config
from enum import IntEnum
#from scipy.signal import butter, lfilter, freqz

from datagenerator import recordRotations, L, N, plotRecordedData


parser = argparse.ArgumentParser()
parser.add_argument("--steps", type=int, default=15, help="steps to run")
parser.add_argument("--debug", type=bool, default=False, help="Use debug mode")
parser.add_argument("--use_sim", type=bool, default=False, help="Use simulator for data generation")
args = parser.parse_args()

if args.use_sim:
    from runsim import collectData


# In loss, use -train_target_signal for compensating signal,
# Use positive for making clone

# Define enum, so it is easy to update meaning of data indices
class Channel(IntEnum): # Channels in data block
    ANGLE = 0
    SIGNAL1 = 1 # torque / speed


def moving_average(a, n=3) :
    ret = np.cumsum(a, dtype=float)
    ret[n:] = ret[n:] - ret[:-n]
    return ret[n - 1:] / n

# This model tries to learn a periodical signal from provided input data.
# After learning, the model can predict future values of similar signal.
# Generate_data.py can be used to generate input data for the model.

# number of recordings x rotations x signal length
# Saved data: [angle, signal1]


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
        self.batchnorm = nn.BatchNorm1d(75)

        print("Using %d hidden layers..." % hidden)

    # NN structure follows pytorch sequence example
    def forward(self, input):
        x = F.relu(self.conv1(input))
        x = self.batchnorm(x)
        x = self.linear1(x)
        x = self.flatten(x)
        x = self.linear2(x) # [10, 999]
        output = x

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

    # One step forward shift for signals
    # This should be tested...
    def shift(self, new_tensor, old_tensor):
        # Replace old input values with shifted signal data; old_tensor cannot be overwritten directly!
        tensor = old_tensor.clone() # keep graph
        tensor[:, Channel.SIGNAL1, :] = new_tensor[:, :] # shift signal (new values are in new_tensor, just assign). Currently mby wrong!!!
        tensor[:, Channel.ANGLE, :-1] = old_tensor[:, Channel.ANGLE, 1:] # shift one forward
        return tensor

    # In prediction, we propagate NN ONCE.
    def predict(self, test_input, test_target=None):
            with torch.no_grad(): # Do not update network when predicting
                out = self.seq(test_input)
                if test_target is not None:
                    shift = test_target.size(2)
                    test_target_signal = test_target[:, Channel.SIGNAL1, :shift]
                    loss = self.criterion(out[:, :shift], -test_target_signal) # This makes no sense
                    print("prediction loss:", loss.item())
                out = self.shift(out, test_input) # Combine angle and signal again
                y = out.detach().numpy()
            #print("y:", y.shape)
            return y #[:, 0] # return the 'new' prediction value


    def train(self, train_input, train_target):
        def closure():
            self.optimizer.zero_grad()
            out = self.seq(train_input)
            #print("[train] out:", out.size(), "target:", train_target.size())
            shift = train_target.size(2)
            train_target_signal = train_target[:, Channel.SIGNAL1, :shift]
            out = out[:, :shift]
            loss = self.criterion(out, -train_target_signal) # flip to get compensation loss. (-train)
            plot2(out, "predictions/train_prediction.svg")

            print("loss:", loss.item())
            loss.backward()
            return loss
        self.optimizer.step(closure)

def plot(input_data, output, iteration):
    print("input_data shape:", input_data.shape)
    print("output_data shape:", output.shape)
    #print("tweaked output shape:", output[0, 0, -999:])

    input_length = input_data.size(2)
    plt.figure(figsize=(15,7))
    plt.xlabel(r"Angle ($\theta_m$)", fontsize=18, labelpad=5)
    plt.ylabel("Torque [pu.]", fontsize=18, labelpad=5)
    plt.xticks(fontsize=18)
    plt.yticks(fontsize=18)
    x1 = np.arange(input_length) #input_length

    x2 = np.arange(input_length, input_length + output.shape[1])
    plt.plot(x1, input_data[0, 1, :])
    plt.plot(x2, output[0, :])

    plt.savefig("predictions/prediction%d.svg" % iteration)
    plt.close()

def getDataBatch():
    # Gather a data batch with N-rows

    # batch_num x signals_num x signal_len
    data = np.empty((N, 2, L), 'float64')
    print("data shape:", data.shape)

    for i in range(0, N):
        print("Collecting sample: {}/{}".format((i+1), N))
        if args.use_sim:
            data[i, :, :] = collectData(rotations=config.repetitions)
        else:    
            data[i, :, :] = recordRotations(rotations=config.repetitions)
        
    if args.debug:
        plotRecordedData(data)

    return data


def main(args):

    if not args.debug:
        matplotlib.use("Agg")

    if args.use_sim:
        from runsim import collectData


    # Create a new model
    model = Model()

    for i in range(args.steps):
        print("STEP:", i)

        # Get a batch of data
        data = getDataBatch()
        
        test_input = torch.from_numpy(data[..., :-1] )
        test_target = torch.from_numpy(data[..., 1:]) # match only signal values (ignore angle)
        train_input = torch.from_numpy(data[..., :-1])
        train_target = torch.from_numpy(data[..., 1:])

        # 1) Let the model learn
        model.train(train_input, train_target)
        plot3(train_input, "predictions/train_input.svg")

        # 2) Check how model is performing
        y = model.predict(test_input, test_target)
        #plot2(y[:, 1, :], "predictions/test_prediction.svg")
        #plot3(test_input, "predictions/test_input.svg")

        # 3) Visualize performance
        plot(test_input, y[:, 1, :], i)
        #plot2(y, i)


    # We know that pulsation pattern should be relatively smooth.
    # Filter out the noise with low pass filter. This is better to do manually,
    # Since we already know what we want.
    angle = moving_average(y[0, 0, :], n=1)
    signal = moving_average(y[0, 1, :], n=1)
    plt.plot(np.arange(len(signal)), signal, color='green')
    #plt.plot(np.arange(len(signal)), angle, color='blue')
    plt.savefig("predictions/filetered.svg")
    # Save result
    torch.save(model.seq.state_dict(), "predictions/compensator.mdl")
    np.savetxt("predictions/compensation-pattern.csv", np.array([angle, signal]), delimiter=",")

if __name__ == "__main__":
    main(args)
