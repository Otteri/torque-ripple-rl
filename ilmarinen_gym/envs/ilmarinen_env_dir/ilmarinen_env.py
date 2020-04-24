from python_interface import Ilmarinen
import matplotlib.pyplot as plt
import numpy as np
import gym

class IlmarinenEnv(gym.Env):
    sim = None
    step_num = 0
    positive_reward_combo = 0
    signal_prev = 0

    def __init__(self, visual=False, compensation=True):
        self.sim = Ilmarinen.SandboxApi()
        self.compensation = compensation
        self.step_size = 0.0001
        self.max_steps = 5000 # 0.001 * 1000 = 1 (s)

        # self.rpm = 
        # revolution_in_steps = (rpm / 60s) / step_size
        self.revolution_in_steps = (60 / 60) / self.step_size

        # For rendering:
        if visual:
            self.compensation_torque = np.zeros(self.max_steps+1) # compensation
            self.actual_torque = np.zeros(self.max_steps+1) # cogging
            self.time = np.arange(0, (self.max_steps+1) * self.step_size, self.step_size)
            plt.ion()
            self.figure = plt.figure()
            self.axle = self.figure.add_subplot(111)
            self.line1, = self.axle.plot(self.time, self.compensation_torque, 
                color='darkorange', label='Compensation', linestyle='-', linewidth=0.5)
            self.line2, = self.axle.plot(self.time, self.actual_torque, 
                color='red', label="Actual", linestyle='-', linewidth=0.5)
            plt.ylabel("Amplitude (p.u.)")
            plt.xlabel("Time (s)")
            plt.title("Torque plot")
            plt.legend()
            axes = plt.gca()
            axes.set_ylim([-1.0, 1.0])

    def step(self, value):     
        # The aim is to compensate disturbances, so we want the signal to
        # stay steady. Give positive reward if the value stays ~constant.
        # If the difference is huge, then give negative reward.

        def getReward(reference, actual):
            difference = abs(reference - actual)
            reward = -difference # Use negative cost as reward:
            return reward

        # Done if some of the following conditions have occured:
        # 1) Max step number has been reached
        # 2) Fault has occured. Cannot continue
        # 3) Problem is solved if one full revolution with positive reward only
        def getDone():
            done = self.step_num+1 >= self.max_steps \
                   or self.sim.signal.getTripCode() != -1 #\
                   #or self.positive_reward_combo >= self.revolution_in_steps
            return done

        if self.compensation:
            self.sim.signal.setCompensationTorque(float(value))
        self.sim.command.step(0.01)
        response_signal = self.sim.signal.gt_FA2TC_data_().n_meas
        reward = getReward(self.sim.signal.getSpeedReference(), response_signal)
        done = getDone()
        info = self.sim.signal.getTripCode()
        torque = self.sim.signal.getSimActualTorqueFiltered()
        state = np.array([torque, self.sim.signal.getRotorElectricalAngle()])
        self.step_num += 1

        return state, reward, done, info

    def reset(self):
        del self.sim
        self.step_num = 0
        self.last_compensation_value = 0
        self.sim = Ilmarinen.SandboxApi()
        randomization = np.random.uniform(0.0, 1.0)
        self.sim.command.step(10 + randomization) # run past startup
        signal = self.sim.signal.getSimActualTorqueFiltered()
        angle = self.sim.signal.getRotorElectricalAngle()
        return np.array([signal, angle])

    def render(self):
        self.compensation_torque[self.step_num] = self.sim.signal.getCompensationTorque()
        self.line1.set_ydata(self.compensation_torque)

        self.actual_torque[self.step_num] = self.sim.signal.getSimActualTorqueFiltered()
        self.line2.set_ydata(self.actual_torque)

        self.figure.canvas.draw()
        self.figure.canvas.flush_events()
