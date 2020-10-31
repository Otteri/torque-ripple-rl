import numpy as np
import matplotlib
import matplotlib.pyplot as plt
import config

# Briefly:
# Generates two channel data: [signal, angle]
# Multiple samples can be recorded at once. The output data will
# be stacked, so the final output is three dimensional array.
# The module can also visualize the generated data.
# Just call recordRotations function with desired arguments.

# ==============================================================
# Settings
L = config.L
N = config.N
step_size = config.step_size
repetitions = config.repetitions
harmonics = config.harmonics
PI2 = 2*np.pi

# ==============================================================
# Data generation logic

def _sample(angle):
    torque = 0.0
    for harmonic_n in harmonics:
        magnitue = harmonics[harmonic_n]
        torque += magnitue * np.cos(harmonic_n * angle)
    return torque

# Returns list including rotations
# If default one rotation, then list has only one item
# Currently only considers single rotation
def _recordRotation():
    signal_data = np.empty(L, 'float64')
    rotation_data = np.empty(L, 'float64')
    #step_size = PI2 / L # divide one revolution to L steps (theoretical value)
    requested_sample_amount = L
    current_sample_num = 0

    # Do one approximately full mechanical rotation
    # Might not be precisely full rotation, due to added noise
    rotor_angle = np.random.uniform(0, PI2) # Start from random angle
    while(current_sample_num < requested_sample_amount):
        rotation_data[current_sample_num] = rotor_angle
        signal_data[current_sample_num] = _sample(rotor_angle)

        noise = np.random.uniform(-config.noise, config.noise) # simulate encoder noise (%)
        rotor_angle += step_size + noise
        current_sample_num += 1

        # Make angle to stay within limits: [0, PI2]
        if (rotor_angle >= PI2):
            rotor_angle = 0

    recorded_data = np.vstack((rotation_data, signal_data)) # [angle, signal1]
    return recorded_data # [angle, signal1]

def recordRotations(rotations=1, viz=False):
    data = np.empty((rotations, 2, L), 'float64')
    for i in range(0, rotations):
        print("Collecting sample: {}/{}".format((i+1), rotations))
        data[i, :, :] = _recordRotation()

    # Show recorded samples if user wants to see them.
    if viz:
        _plotRecordedData(data)

    return data

# ==============================================================
# Data visualization logic below

def _keypress(event, ax1, ax2):
    ax1.clear()
    ax2.clear()
    # TODO: Do not clear whole plot & texts (clf didn't work)
    ax1.title.set_text('Rotor Angle [rad]')
    ax2.title.set_text('Signal 1: torque [Nm]')

def _plot(input_vector, hndl, color='b'):
    input_length = input_vector.shape[0]
    x = np.arange(input_length)
    hndl.plot(x, input_vector, color, linewidth=2.0)
    hndl.plot(x, input_vector, color, linewidth=2.0)

def _plotRecordedData(recorded_data):
    data = recorded_data

    # Plot settings
    fig, (ax1, ax2) = plt.subplots(2, 1)
    fig.canvas.mpl_connect("key_press_event", lambda event: _keypress(event, ax1, ax2))
    ax1.title.set_text('Rotor Angle [rad]')
    ax2.title.set_text('Signal 1: torque [Nm]')
    fig.tight_layout()
    print("Mouse click: plot over. Keyboard key: clear and plot next.")

    for i in range(0, data.shape[0]):
        print("Showing input: {}/{}".format((i+1), data.shape[0]))
        _plot(data[i, 1, :], ax2, 'b') # signal1
        _plot(data[i, 0, :], ax1, 'r') # angle
        plt.waitforbuttonpress()

    plt.close()
