import sys
import time
import argparse
from train import main
import torch.multiprocessing as mp
import matplotlib.pyplot as plt
import numpy as np

parser = argparse.ArgumentParser()
parser.add_argument("--episodes", "-ep", type=int, default=500, help="Number of episodes to train for")
parser.add_argument("--render", action='store_true', help="Render compensation waveform")
parser.add_argument("--test", "-t", type=str, default=None, help="Model to be tested")
parser.add_argument("--load", type=str, default=None, help="Load model and continue training")
parser.add_argument("--es", type=str, default=None, help="v1,v2,v3...")
parser.add_argument("--gammas", type=str, default=None, help="v1,v2,v3...")
parser.add_argument("--alphas", type=str, default=None,  help="v1,v2,v3...")
# Example call $ python multirunner.py --gammas 0.2,0.4,0.6,0.8,0.87,0.95

num_processes = 6
color1 = '#1d9bf0'
color2 = '#43ccd9'
color3 = '#a0eb9f'
color4 = '#e0ce75'
color5 = '#fa9e55'
color6 = '#ff4926'
colors = [color1, color2, color3, color4, color5, color6]
labels = ['0.20', '0.40', '0.60', '0.80', '0.87', '0.95'] # gammas
#labels = ['0.10', '0.20', '0.30', '0.50', '0.70', '0.90'] # alphas
#labels = ['50', '100', '200', '500', '750', '1000'] #es

# Convert comma separated string to array of floats
def getParameterValues(parameters):
    params = parameters.split(',')
    params = [float(param) for param in params] # str to float
    return params

def run(args):

    # give some defaults that can be overwritten soon
    alphas = [0.6] * num_processes
    gammas = [0.6] * num_processes
    es = [100] * num_processes
    if args.gammas:
        gammas = getParameterValues(args.gammas)
        print("gammas:", gammas)

    if args.alphas:
        alphas = getParameterValues(args.alphas)
        print("alphas:", alphas)

    if args.es:
        es = getParameterValues(args.es)
        print("es:", es)


    fig, ax = plt.subplots(figsize=(6.4,5.0), dpi=100)

    processes = []
    manager = mp.Manager()
    return_dict = manager.dict()
    for i in range(0, num_processes):
        p = mp.Process(target=main, args=(i, alphas[i], gammas[i], es[i], return_dict))
        p.start()
        processes.append(p)
        time.sleep(0.2) # give time to read dat-files
    for p in processes:
        p.join()
    print("returned:", return_dict.keys())

    for i in range(6):
        x = np.array(return_dict['iterations' + str(i)])
        y = np.array(return_dict['rewards' + str(i)])
        x = x[350:]
        y = y[350:]
        plt.plot(x, y, label=labels[i], linewidth=0.8, color=colors[i])

    plt.legend(title='Gamma ($\gamma$)', loc='lower right', fancybox=True) # for gammas
    #plt.legend(title='$\\alpha$', loc='lower right', fancybox=True) # for alphas
    #plt.legend(title='$k_\epsilon$', loc='lower right', fancybox=True) # for alphas
    plt.ylabel("Reward", fontsize=12)
    plt.xlabel("Rotation no.", fontsize=12)

    # harcoded for 60 rpm, fs = 0.01, 300s data
    # 300s * 60 rpm = 18000 rev in total.
    iterations = return_dict['iterations0']
    ticks = iterations[::100] # 0.01 -> 1s
    print("iterations:", len(iterations))
    ticks = np.arange(0, 18000, step=1800)
    plt.xticks(iterations[::2897], ticks)
    ax.grid(color='white')
    ax.set_facecolor('lightblue')
    ax.patch.set_alpha(.20)

    plt.grid(True)
    plt.show(block=True)

if __name__ == '__main__':
    args = parser.parse_args()
    run(args)