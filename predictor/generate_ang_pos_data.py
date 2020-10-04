import numpy as np
import torch
import config

# Generates two channel data
# [signal; rotor angle]

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
# Currently only considers single rotation
def recordRotations(rotations=1):
    rotations = 1
    signal_data = np.empty(L, 'float64')
    rotation_data = np.empty(L, 'float64')
    step_size = PI2 / L # divide one revolution to L steps (theoretical value)
    requested_sample_amount = L
    current_sample_num = 0

    # Do one approximately full mechanical rotation
    # Might not be precisely full rotation, due to added noise
    rotor_angle = np.random.uniform(0, PI2) # Start from random angle
    while(current_sample_num < requested_sample_amount):
        rotation_data[current_sample_num] = rotor_angle
        signal_data[current_sample_num] = sample(rotor_angle)

        noise = np.random.uniform(-config.noise, config.noise) # simulate encoder noise (%)
        rotor_angle += step_size + noise
        current_sample_num += 1

        # Make angle to stay within limits: [0, PI2]
        if (rotor_angle >= PI2):
            rotor_angle = 0

    recorded_data = np.vstack((rotation_data, signal_data)) # [angle, signal1]
    return recorded_data # [angle, signal1]

def main():
    # batch_num x signals_num x signal_len
    data = np.empty((N, 2, L), 'float64')
    print("data shape:", data.shape)

    # Collect N vectors
    for i in range(0, N):
        data[i, :, :] = recordRotations(rotations=config.repetitions)

    np.save(config.datafile+".npy", data)
    torch.save(data, open(config.datafile+".pt", 'wb'))

    print("Generation complete. Data shape:", data.shape)

if __name__ == "__main__":
    main()
