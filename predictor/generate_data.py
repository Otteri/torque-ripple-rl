import numpy as np
import torch
import config

L = config.L
N = config.N
repetitions = config.repetitions
harmonics = config.harmonics

def sample(angle):
    torque = 0.0
    for harmonic_n in harmonics:
        magnitue = harmonics[harmonic_n]
        torque += magnitue * np.cos(harmonic_n * angle)
    return torque

def recordRotation(step_size):
    signal = np.empty(L, 'float64')
    rotor_angle = np.random.uniform(0, 2*np.pi) # Start from random angle

    # Do one full mechanical rotation
    for j in range(0, L):
        signal[j] = sample(rotor_angle)
        rotor_angle += step_size
    
    return signal

def main():
    data = np.empty((N, L), 'float64')
    step_size = 2*np.pi / (L / repetitions) # divide one revolution to L steps

    # Collect N vectors
    for i in range(0, N):
        data[i] = recordRotation(step_size)

    torch.save(data, open(config.datafile, 'wb'))
    print("Generation complete. Data shape:", data.shape)

if __name__ == "__main__":
    main()
