import matplotlib.pyplot as plt
import numpy as np
from gym import Env

class IlmarinenEnv(Env):
    sim = None
    step_num = 0
    angle_prev = 1.0
    prev_actual = 0.0

    last_compensation_value = 0
    # Step size has to be enough small to be able to utilize this function,
    # otherwise the finish signal can come too late.
    def hasPeriodFinished(self):
        MAX_ANGLE = 1.0
        current_angle = self.sim.signal.getRotorElectricalAngleCtrl()
        expected_change = current_angle - self.angle_prev
        self.angle_prev = current_angle

        # It is important to return the True a fraction-step before the rotation
        # Because first change calculation is incorrect after rotation without
        # special handling. (Angle is nonlinear function).
        if (current_angle + expected_change) > MAX_ANGLE or \
            self.step_num >= self.max_steps-1: # don't allow buffer overflow
            return True
        return False


    # Zeroes signals
    def clearPlot(self):
        self.compensation_torque = np.zeros(self.max_steps+1) # compensation
        self.actual_torque = np.zeros(self.max_steps+1) # cogging
        self.rotor_angle = np.zeros(self.max_steps+1)
        self.speed = np.zeros(self.max_steps+1)

    def __init__(self, visual=False, compensation=True):
        from ilmarinen.envs.v2 import Ilmarinen as Ilmarinen_v2 # here to avoid name collision
        self.sim = Ilmarinen_v2.SandboxApi()
        self.compensation = compensation
        self.step_size = 0.001
        self.max_steps = 251
        self.rpm = 60

        # Set Sim into learning mode:
        self.sim.command.toggleQlr(True)
        self.sim.command.toggleLearning(True)
        self.revolution_in_steps = (self.rpm / 60) / self.step_size

        # Run to stable and then beginning of a electrical period
        self.sim.command.step(30)
        while not self.hasPeriodFinished():
            self.sim.command.step(0.001)

        # For rendering:
        if visual:
            self.clearPlot() # use to allocate memory
            self.time = np.arange(0, (self.max_steps+1) * self.step_size, self.step_size)
            plt.ion()
            self.figure = plt.figure()
            self.axle = self.figure.add_subplot(111)
            self.line1, = self.axle.plot(self.time, self.compensation_torque, 
                color='yellowgreen', label='Compensation torque', linestyle='-', linewidth=0.5)
            self.line2, = self.axle.plot(self.time, self.actual_torque, 
                color='red', label="Actual torque", linestyle='-', linewidth=0.5)
            plt.ylabel("Amplitude [pu.]")
            plt.xlabel("Time [s]")
            plt.legend(loc='upper right')
            axes = plt.gca()
            axes.set_ylim([-0.70, 0.70])

    def step(self, value):
        # The aim is to compensate disturbances,
        # so we want the signal to stay steady:
        def getReward(reference, actual):
            self.prev_actual = actual
            difference = abs(reference - actual)
            cost = difference
            return -cost

        # Done if some of the following conditions have occured:
        # 1) Max step number has been reached
        # 2) Fault has occured. Cannot continue
        # 3) Problem is solved if one full revolution with positive reward only
        def getDone():
            done = self.hasPeriodFinished()
            return done

        if self.compensation:
            self.sim.signal.setAction(value)
            self.last_compensation_value = value

        self.sim.command.step(self.step_size)
        reward = getReward(0, self.sim.signal.getSimActualTorque())
        done = getDone()
        info = self.sim.signal.getTripCode()
        torque = self.sim.signal.getSimActualTorqueFiltered()
        state = np.array([torque, self.sim.signal.getRotorElectricalAngleCtrl(), 0])
        self.step_num += 1

        return state, reward, done, info

    # Fast training: do not actually reset the whole sim
    # New period has started. Just reset steps.
    def reset(self):
        self.step_num = 0
        signal = self.sim.signal.getSimActualTorqueFiltered()
        angle = self.sim.signal.getRotorElectricalAngleCtrl()
        self.last_compensation_value = 0
        return np.array([signal, angle, 0])

        # HardReset -- use when trips. Restarts whole simulator
    def hardReset(self):
        del self.sim
        self.clearPlot()
        from ilmarinen.envs.v2 import Ilmarinen as Ilmarinen_v2
        self.sim = Ilmarinen_v2.SandboxApi()

        # Run to stable and then beginning of a electrical period
        self.sim.command.step(30)
        while not self.hasPeriodFinished():
            self.sim.command.step(0.001)

        return self.reset()

    def render(self):
        self.compensation_torque[self.step_num] = self.sim.signal.getAction()
        self.actual_torque[self.step_num] = self.sim.signal.getSimActualTorque()
        
        self.line1.set_ydata(self.compensation_torque)
        self.line2.set_ydata(self.actual_torque)

        self.figure.canvas.draw()
        self.figure.canvas.flush_events()

# Good to know:
# The period length is not fixed, because rotor rotation speed is not constant, due to
# disturbances and applied compensation torque, which creates variaton to rotation speed. 
# Therefore, it is possible to overrun constant size buffer. For this reason hasPeriodFinished
# function checks that buffers are not getting overrun, and in case of exceeding buffers, a new 
# period is started. This must be accounted when plotting the results. Some periods can be just
# a few steps long and these should be ignored. Todo: get rid of this nuisance.
