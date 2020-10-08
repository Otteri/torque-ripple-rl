###################
#      Model      #
###################
learning_rate = 0.1       # Optimizer lr
hidden_layers = 32       # Amount of neurons in hidden layers


###################
# Data generation #
###################
L = 1000                  # Signal length
N = 10                    # Number of signals in the data matrix
repetitions = 30          # How many times pattern is repeated during single period
datafile = "traindata"    # Where generated data is saved

harmonics = {             # Signal shape.
    1 : 0.5,
    2 : 0.3
}

noise = 0.04              # Makes the signal jagged. 0.02 = 2% error.

step_size = 0.001        # Set step size: PI2 / L for one data revolution
###################
#    Plotting     #
###################
color = 'b'
dpi = 100
