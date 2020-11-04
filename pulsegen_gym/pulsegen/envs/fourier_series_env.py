import matplotlib.pyplot as plt
import numpy as np
from gym import Env

# Briefly:
# Generates two channel data: [signal, angle]
# Multiple samples can be recorded at once. The output data will
# be stacked, so the final output is three dimensional array.
# The module can also visualize the generated data.
# Just call recordRotations function with desired arguments.

PI2 = 2*np.pi
# ==============================================================
# Data generation logic

class FourierSeries(Env):

    def __init__(self, config_path=None):
        if config_path:
            import sys
            print("Loading config from:", config_path)
            sys.path.append(config_path)
            import config
        else:
            from . import default_config as config

        self.data_length = config.L
        self.step_size = config.step_size
        self.harmonics = config.harmonics
        self.noise = config.noise

    def _sample(self, angle):
        torque = 0.0
        for harmonic_n in self.harmonics:
            magnitue = self.harmonics[harmonic_n]
            torque += magnitue * np.cos(harmonic_n * angle)
        return torque

    # Returns list including rotations
    # If default one rotation, then list has only one item
    # Currently only considers single rotation
    def _recordRotation(self):
        signal_data = np.empty(self.data_length, 'float64')
        rotation_data = np.empty(self.data_length, 'float64')
        current_sample_num = 0

        # Do one approximately full mechanical rotation
        # Might not be precisely full rotation, due to added noise
        rotor_angle = np.random.uniform(0, PI2) # Start from random angle
        while(current_sample_num < self.data_length):
            rotation_data[current_sample_num] = rotor_angle
            signal_data[current_sample_num] = self._sample(rotor_angle)

            noise = np.random.uniform(-self.noise, self.noise) # simulate encoder noise (%)
            rotor_angle += self.step_size + noise
            current_sample_num += 1

            # Make angle to stay within limits: [0, PI2]
            if (rotor_angle >= PI2):
                rotor_angle = 0

        recorded_data = np.vstack((rotation_data, signal_data)) # [angle, signal1]
        return recorded_data # [angle, signal1]

    def recordRotations(self, rotations=1, viz=False):
        data = np.empty((rotations, 2, self.data_length), 'float64')
        for i in range(0, rotations):
            print("Collecting sample: {}/{}".format((i+1), rotations))
            data[i, :, :] = self._recordRotation()

        # Show recorded samples if user wants to see them.
        if viz:
            self._plotRecordedData(data)

        return data

# ==============================================================
# Data visualization logic below

    def _keypress(self, event, ax1, ax2):
        ax1.clear()
        ax2.clear()
        # TODO: Do not clear whole plot & texts (clf didn't work)
        ax1.title.set_text('Rotor Angle [rad]')
        ax2.title.set_text('Signal 1: torque [Nm]')

    def _plot(self, input_vector, hndl, color='b'):
        input_length = input_vector.shape[0]
        x = np.arange(input_length)
        hndl.plot(x, input_vector, color, linewidth=2.0)
        hndl.plot(x, input_vector, color, linewidth=2.0)

    def _plotRecordedData(self, recorded_data):
        data = recorded_data

        # Plot settings
        fig, (ax1, ax2) = plt.subplots(2, 1)
        fig.canvas.mpl_connect("key_press_event", lambda event: self._keypress(event, ax1, ax2))
        ax1.title.set_text('Rotor Angle [rad]')
        ax2.title.set_text('Signal 1: torque [Nm]')
        fig.tight_layout()
        print("Mouse click: plot over. Keyboard key: clear and plot next.")

        for i in range(0, data.shape[0]):
            print("Showing input: {}/{}".format((i+1), data.shape[0]))
            self._plot(data[i, 1, :], ax2, 'b') # signal1
            self._plot(data[i, 0, :], ax1, 'r') # angleS
            plt.waitforbuttonpress()

        plt.close()

    def step(self):
        print("Not implemented yet")
    def reset(self):
        print("Not implemented yet")
    def render(self):
        print("Not implemented yet")
