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
    input_data = torch.from_numpy(data[0:, :-1])

    # Define where data will be drawn
    fig = plt.figure(figsize=(15,5), dpi=config.dpi)
    fig.canvas.mpl_connect("key_press_event", lambda event: keypress(event, fig))
    print("Mouse click: plot over. Keyboard key: clear and plot next.")

    # Loop through data
    for i, input_vector in enumerate(input_data, start=1):
        print("Showing input: {}/{}".format(i, len(input_data)))
        plot(input_vector)
        plt.waitforbuttonpress()   

if __name__ == "__main__":
    main()
