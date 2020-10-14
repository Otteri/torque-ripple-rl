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

from datagenerator import recordRotations, L, N, plotRecordedData

# TODO train using speed signal

parser = argparse.ArgumentParser()
parser.add_argument("--steps", type=int, default=15, help="steps to run")
parser.add_argument("--debug", default=False, action="store_true", help="Use debug mode")
parser.add_argument("--invert", default=False, action="store_true", help="Invert learning outcome")
parser.add_argument("--use_sim", type=bool, default=False, help="Use simulator for data generation")
args = parser.parse_args()

if args.use_sim:
    from runsim import collectData

# Define enum, so it is easy to update meaning of data indices
class Channel(IntEnum): # Channels in data block
    ANGLE = 0
    SIGNAL1 = 1 # torque / speed

ave_n = 5

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
        self.linear1 = nn.Linear(166, 40)
        self.linear2 = nn.Linear(3000, 999)
        self.conv1 = nn.Conv1d(2, hidden, 3, stride=2, padding=2)
        self.avg_pool = nn.AvgPool1d(5, stride=3)
        self.flatten = Flatten()
        self.batchnorm = nn.BatchNorm1d(75)

        print("Using %d hidden layers..." % hidden)

    # NN structure follows pytorch sequence example
    def forward(self, input):
        x = self.avg_pool(F.relu(self.conv1(input)))
        x = self.batchnorm(x)
        x = self.linear1(x)
        x = self.flatten(x)
        x = self.linear2(x)
        return x

class Model(object):
    def __init__(self):
        self.seq = Sequence(75) #config.hidden_layers = N or repetitions
        self.seq.double()
        self.criterion = nn.MSELoss()
        # use LBFGS as optimizer since we can load the whole data to train
        self.optimizer = optim.LBFGS(self.seq.parameters(), lr=config.learning_rate)

    # One step forward shift for signals
    def shift(self, new_tensor, old_tensor):
        # Replace old input values with shifted signal data; old_tensor cannot be overwritten directly!
        tensor = old_tensor.clone() # keep graph
        tensor[:, Channel.SIGNAL1, :] = new_tensor[:, :] # shift signal (new values are in new_tensor, just assign). Currently mby wrong!!!
        tensor[:, Channel.ANGLE, :-1] = old_tensor[:, Channel.ANGLE, 1:] # shift one forward
        return tensor

    # Avg-filter input signal, since it can be quite noisy and we don't want to try learn white noise.
    # Add padding by copying first few values in the beginning, so the data vector length does not change.
    def preprocessBatch(self, input_data, n=1):
        filtered_data = input_data.clone()
        for i in range(0, input_data.size(1)):
            padded_input_data = torch.cat((input_data[i, Channel.SIGNAL1, 0:(n-1)], input_data[i, Channel.SIGNAL1, :]))
            filtered_data[i, Channel.SIGNAL1, :] = moving_average(padded_input_data, n=n)
        return filtered_data

    # In prediction, we propagate NN ONCE.
    def predict(self, test_input, test_target=None):
            with torch.no_grad(): # Do not update network when predicting
                filtered_input = self.preprocessBatch(test_input, n=ave_n)
                filtered_target = self.preprocessBatch(test_target, n=ave_n)
                out = self.seq(filtered_input)
                if test_target is not None:
                    shift = test_target.size(2)
                    test_target_signal = filtered_target[:, Channel.SIGNAL1, :shift]
                    if args.invert:
                        loss = self.criterion(out[:, :shift], test_target_signal) # Easier to compare input
                    else:
                        loss = self.criterion(out[:, :shift], -test_target_signal) # Learn compensation signal

                    print("prediction loss:", loss.item())
                out = self.shift(out, test_input) # Combine angle and signal again; use original input data
                y = out.detach().numpy()
            return y, filtered_input #[:, 0] # return the 'new' prediction value


    def train(self, train_input, train_target):
        def closure():
            self.optimizer.zero_grad()
            filtered_input = self.preprocessBatch(train_input, n=ave_n)
            filtered_target = self.preprocessBatch(train_target, n=ave_n)
            out = self.seq(filtered_input)
            shift = train_target.size(2)
            train_target_signal = filtered_target[:, Channel.SIGNAL1, :shift]
            out = out[:, :shift]
            if args.invert:
                loss = self.criterion(out, train_target_signal)
            else:
                loss = self.criterion(out, -train_target_signal)

            print("loss:", loss.item())
            loss.backward()
            return loss
        self.optimizer.step(closure)

def plot(input_data, filtered_input, output, iteration):
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

    x2 = np.arange(input_length, input_length + output.shape[1]) # Use to plot next
    plt.plot(x1, input_data[0, 1, :], "-", color='b', label="input")
    plt.plot(x1, filtered_input[0, 1, :], '--', color='r', label="filtered input")
    plt.plot(x1, output[0, :], '-', color='green', label="learning output")
    plt.legend()

    plt.savefig("predictions/prediction%d.svg" % iteration)
    plt.close()


# Gather a data batch with N-rows
# batch_num x signals_num x signal_len
def getDataBatch():
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

    # Configure
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

        # 2) Check how model is performing
        y, filtered_input = model.predict(test_input, test_target)

        # 3) Visualize performance
        plot(test_input, filtered_input, y[:, 1, :], i)


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
