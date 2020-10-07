import argparse
import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import config

# This model tries to learn a periodical signal from provided input data.
# After learning, the model can predict future values of similar signal.
# Generate_data.py can be used to generate input data for the model.

class Sequence(nn.Module):
    def __init__(self, hidden=51):
        super(Sequence, self).__init__()
        self.hidden = hidden
        self.linear1 = nn.Linear(1, hidden) # bias layer to create offset from 0-torq
        self.lstm1 = nn.LSTMCell(hidden, hidden)
        self.lstm2 = nn.LSTMCell(hidden, hidden)
        self.lstm3 = nn.LSTMCell(hidden, hidden)
        self.linear2 = nn.Linear(hidden, 1) # scaler layer (may be useless)
        print("Using %d hidden layers..." % hidden)

    # NN structure follows pytorch sequence example
    def forward(self, input, future = 0):
        outputs = []
        h_t = torch.zeros(input.size(0), self.hidden, dtype=torch.double)
        c_t = torch.zeros(input.size(0), self.hidden, dtype=torch.double)
        h_t2 = torch.zeros(input.size(0), self.hidden, dtype=torch.double)
        c_t2 = torch.zeros(input.size(0), self.hidden, dtype=torch.double)
        h_t3 = torch.zeros(input.size(0), self.hidden, dtype=torch.double)
        c_t3 = torch.zeros(input.size(0), self.hidden, dtype=torch.double)


        for i, input_t in enumerate(input.chunk(input.size(1), dim=1)):
            input_t = self.linear1(input_t)
            h_t, c_t = self.lstm1(input_t, (h_t, c_t))
            h_t2, c_t2 = self.lstm2(h_t, (h_t2, c_t2))
            h_t3, c_t3 = self.lstm3(h_t2, (h_t3, c_t3))
            output = self.linear2(h_t3)
            outputs += [output]
        for i in range(future):# if we should predict the future
            print("output:", output)
            print("out shape:", output.shape)
            output = self.linear1(output)
            h_t, c_t = self.lstm1(output, (h_t, c_t))
            h_t2, c_t2 = self.lstm2(h_t, (h_t2, c_t2))
            h_t3, c_t3 = self.lstm3(h_t2, (h_t3, c_t3))
            output = self.linear2(h_t3)
            outputs += [output]
        outputs = torch.stack(outputs, 1).squeeze(2)
        return outputs

class Model(object):
    def __init__(self):
        self.seq = Sequence(config.hidden_layers)
        self.seq.double()
        self.criterion = nn.MSELoss()
        # use LBFGS as optimizer since we can load the whole data to train
        self.optimizer = optim.LBFGS(self.seq.parameters(), lr=config.learning_rate)
        self.future = 1000

    def predict(self, test_input, test_target=None):
        with torch.no_grad(): # Do not update network when predicting
            pred = self.seq(test_input, future=self.future)
            if test_target is not None:
                loss = self.criterion(pred[:, :-self.future], test_target)
                print("prediction loss:", loss.item())
            y = pred.detach().numpy()
        return y

    def train(self, train_input, train_target):
        def closure():
            self.optimizer.zero_grad()
            out = self.seq(train_input)
            #print("train_target shape:", train_target.shape)
            #print("out shape:", out.shape)
            loss = self.criterion(out, train_target)
            print("loss:", loss.item())
            loss.backward()
            return loss
        self.optimizer.step(closure)

def plot(input_data, future, output, iteration):
    input_length = input_data.size(1)
    plt.figure(figsize=(15,7))
    plt.xlabel(r"Angle ($\theta_m$)", fontsize=18, labelpad=5)
    plt.ylabel("Torque [pu.]", fontsize=18, labelpad=5)
    plt.xticks(fontsize=18)
    plt.yticks(fontsize=18)
    def draw(yi, color):
        x = np.arange(input_length)
        plt.plot(x, yi[:input_length], color, linewidth = 2.0)
        plt.plot(np.arange(input_length, input_length + future), yi[input_length:], color + ':', linewidth = 2.0)
    draw(output[0], 'b')
    plt.savefig("predictions/prediction%d.svg" % iteration)
    plt.close()

def main(args):
    # Load data.
    # input: x[i], target: x[i+1]
    # One step lag allows guessing and learning.
    # First vector is for testing and all others for training.
    data = torch.load("traindata.pt")
    test_input = torch.from_numpy(data[:1, :-1])
    test_target = torch.from_numpy(data[:1, 1:])
    train_input = torch.from_numpy(data[1:, :-1])
    train_target = torch.from_numpy(data[1:, 1:])

    # Create a new model
    model = Model()

    # Run
    for i in range(args.steps):
        print("STEP:", i)

        # 1) Let the model learn
        model.train(train_input, train_target)

        # 2) Check how model is performing
        y = model.predict(test_input, test_target)

        # 3) Visualize performance
        plot(test_input, model.future, y, i)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--steps", type=int, default=15, help="steps to run")
    args = parser.parse_args()
    main(args)
