###################
# Data generation #
###################

# Training data looks following:
# [[0.1, 0.2, ..., 0.9]
#  [0.3, 0.4, ..., 1.2]
#  [0.5, 0.6, ..., 1.4]
#  [1.0, 1.1, ..., 1.9]]
# Each row-vector is a single training set.
# Matrix has configurable dimensions: N x L
L = 1000                  # Signal length
N = 50                    # Number of signals in data matrix
repetitions = 8           # How many times pattern is repeated during single period
datafile = "traindata.pt" # Where generated data is saved

# Signal shape.
# First number is the harmonic order which has relation to frequency,
# and second number gives the amplitude of the n-th harmonic.
# Having only single harmonic results to plain sine wave.
harmonics = {
    1 : 0.5,
}
# More complicated signal, uncomment to use
# harmonics = {
#     1 : 0.02,
#     2 : 0.0030,
#     6 : 0.006,
#     12: 0.00227,
#     24: 0.00039
# }

###################
#    Plotting     #
###################
color = 'b'
dpi = 100
