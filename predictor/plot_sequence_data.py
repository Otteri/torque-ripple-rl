import sys
import torch
import numpy as np
import matplotlib
import matplotlib.pyplot as plt
import config

def keypress(event, fig):
    fig.clear()

def plot(input_vector):
    input_length = input_vector.size(0)
    x = np.arange(input_length)
    plt.plot(x, input_vector, config.color, linewidth=2.0)

def main():
    # Load input data
    data = torch.load(config.datafile)
    data = torch.from_numpy(data)

    # Define where data will be drawn
    fig = plt.figure(figsize=(15,5), dpi=config.dpi)
    fig.canvas.mpl_connect("key_press_event", lambda event: keypress(event, fig))
    print("Mouse click: plot over. Keyboard key: clear and plot next.")

    print("dim1:", data.size(0))
    for i in range(0, data.size(0)):
        print("Showing sample: {}/{}".format((i+1), data.size(0)))
        for j in range(0, data.size(1)):
            print("  Showing rotation: {}/{}".format((j+1), data.size(1)))
            plot(data[i, j, :])
            plt.waitforbuttonpress()

if __name__ == "__main__":
    main()
