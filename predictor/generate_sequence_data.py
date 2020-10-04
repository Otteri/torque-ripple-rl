import numpy as np
import torch
import config

L = config.L
N = config.N
repetitions = config.repetitions
harmonics = config.harmonics
PI2 = 2*np.pi

def sample(angle):
    torque = 0.0
    for harmonic_n in harmonics:
        magnitue = harmonics[harmonic_n]
        torque += magnitue * np.cos(harmonic_n * angle)
    return torque

# Returns list including rotations
# If default one rotation, then list has only one item
def recordRotations(rotations=1):
    rotation_data = np.empty((rotations, L), 'float64')
    signal_data = np.empty(L, 'float64')
    step_size = PI2 / L # divide one revolution to L steps (theoretical value)

    # Do one approximately full mechanical rotation
    # Might not be precisely full rotation, due to added noise
    for i in range(0, rotations):
        j = 0
        rotor_angle = np.random.uniform(0, PI2) # Start from random angle
        while(j < L):
            signal_data[j] = sample(rotor_angle)
            j = j + 1
            noise = np.random.uniform(-config.noise, config.noise) # simulate encoder noise (%)
            rotor_angle += step_size + noise
        rotation_data[i] = signal_data
    return rotation_data

def main():
    data = np.empty((N, repetitions, L), 'float64')

    # Collect N vectors
    for i in range(0, N):
        data[i] = recordRotations(rotations=config.repetitions)

    torch.save(data, open(config.datafile, 'wb'))
    print("Generation complete. Data shape:", data.shape)

if __name__ == "__main__":
    main()
