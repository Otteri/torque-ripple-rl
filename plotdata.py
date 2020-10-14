import sys
import numpy as np
import matplotlib
import matplotlib.pyplot as plt
import config

def keypress(event, ax1, ax2):
    ax1.clear()
    ax2.clear()
    # TODO: Do not clear whole plot & texts (clf didn't work)
    ax1.title.set_text('Rotor Angle [rad]')
    ax2.title.set_text('Signal 1: torque [Nm]')

def plot(input_vector, hndl, color='b'):
    input_length = input_vector.shape[0]
    x = np.arange(input_length)
    hndl.plot(x, input_vector, color, linewidth=2.0)
    hndl.plot(x, input_vector, color, linewidth=2.0)

def main():
    # Load input data
    data = np.load(config.datafile + ".npy")

    # Plot settings
    fig, (ax1, ax2) = plt.subplots(2, 1)
    fig.canvas.mpl_connect("key_press_event", lambda event: keypress(event, ax1, ax2))
    ax1.title.set_text('Rotor Angle [rad]')
    ax2.title.set_text('Signal 1: torque [Nm]')
    fig.tight_layout()
    print("Mouse click: plot over. Keyboard key: clear and plot next.")

    for i in range(0, data.shape[0]):
        print("Showing input: {}/{}".format((i+1), data.shape[0]))
        plot(data[i, 1, :], ax2, 'b') # signal1
        plot(data[i, 0, :], ax1, 'r') # angle
        plt.waitforbuttonpress()

if __name__ == "__main__":
    main()
