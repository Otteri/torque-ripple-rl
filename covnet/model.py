import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F

import config
from enum import IntEnum
# Define enum, so it is easy to update meaning of data indices
class Channel(IntEnum): # Channels in data block
    ANGLE = 0
    SIGNAL1 = 1 # torque / speed

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

    def forward(self, input):
        x = self.avg_pool(F.relu(self.conv1(input)))
        x = self.batchnorm(x)
        x = self.linear1(x)
        x = self.flatten(x)
        x = self.linear2(x)
        return x

class Model(object):
    def __init__(self, compensate=False):
        self.seq = Sequence(75).double() #config.hidden_layers = N or repetitions
        self.criterion = nn.MSELoss()
        # use LBFGS as optimizer since we can load the whole data to train
        self.optimizer = optim.LBFGS(self.seq.parameters(), lr=config.learning_rate)

        self.train_compensator = compensate

    # One step forward shift for signals
    # Replace old input values with shifted signal data; old_tensor cannot be overwritten directly!
    def shift(self, new_tensor, old_tensor):
        tensor = old_tensor.clone() # keep graph
        tensor[:, Channel.SIGNAL1, :] = new_tensor[:, :]
        tensor[:, Channel.ANGLE, :-1] = old_tensor[:, Channel.ANGLE, 1:] # shift one forward
        return tensor

    def computeLoss(self, filtered_input_data, filtered_target_data, invert=False):
        y = self.seq(filtered_input_data)
        if filtered_target_data is not None:
            shift = filtered_target_data.size(2)
            filtered_target_signal = filtered_target_data[:, Channel.SIGNAL1, :shift]
            if self.train_compensator:
                loss = self.criterion(y[:, :shift], -filtered_target_signal) # Learn compensation signal
            else:
                loss = self.criterion(y[:, :shift], filtered_target_signal) # Easier to compare input
            return loss, y
        return y

    # In prediction, do not update NN-weights
    def predict(self, test_input, test_target=None):
            with torch.no_grad(): # Do not update network when predicting
                if test_target is not None:
                    loss, out = self.computeLoss(test_input, test_target)
                    print("prediction loss:", loss.item())
                else:
                    out = self.computeLoss(test_input, test_target)
                out = self.shift(out, test_input) # Combine angle and signal again; use original input data
                y = out.detach().numpy()
            return y #[:, 0] # return the 'new' prediction value

    def train(self, train_input, train_target):
        def closure():
            self.optimizer.zero_grad()
            loss, out = self.computeLoss(train_input, train_target)
            print("loss:", loss.item())
            loss.backward()
            return loss
        self.optimizer.step(closure)
