from python_interface import Ilmarinen
import matplotlib.pyplot as plt
import numpy as np
import gym

class IlmarinenEnv(gym.Env):
    sim = None
    step_num = 0
    positive_reward_combo = 0
    signal_prev = 0
    angle_prev = 1 # can't know, assume max

    differencies = np.zeros(500)
    integral = 0
    prev_actual = 0

    last_compensation_value = 0
    # TODO: try to learn one amplitude and mirror that to all others 
    # encountering on a elec period.

    # Step size has to be enough small in order to utilize this function!
    # With large steps the finish signal will come too late.
    def hasPeriodFinished(self):
        MAX_ANGLE = 1.0
        MIN_ANGLE = 0.0
        current_angle = self.sim.signal.getRotorElectricalAngleCtrl()
        expected_change = current_angle - self.angle_prev
        self.angle_prev = current_angle

        # It is important to return the True a fraction-step before the rotation
        # Because first change calculation is incorrect after rotation without
        # special handling. (Angle is nonlinear function).
        if MAX_ANGLE < (current_angle + expected_change): #and \
           #MIN_ANGLE > (current_angle + expected_change):
            return True
        return False


    # Zeroes signals
    def clearPlot(self):
        self.compensation_torque = np.zeros(self.max_steps+1) # compensation
        self.actual_torque = np.zeros(self.max_steps+1) # cogging
        self.rotor_angle = np.zeros(self.max_steps+1)
        self.speed = np.zeros(self.max_steps+1)

    def __init__(self, visual=False, compensation=True):
        self.sim = Ilmarinen.SandboxApi()
        self.compensation = compensation # why does variation imrpove the score?
        self.step_size = 0.001 # + np.random.uniform(0.0, 0.005) # can see up to 100hz; can compensate 24 * 4 = 96
        self.max_steps = 2051
        # 0.001 * 1000 = 1 (s);  60 / 60 = 1 rev/s mech

        # Set Sim into learning mode:
        self.sim.command.toggleQlr(True)
        self.sim.command.toggleLearning(True)

        # self.rpm = 
        # revolution_in_steps = (rpm / 60s) / step_size
        self.revolution_in_steps = (60 / 60) / self.step_size

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
                color='yellowgreen', label='Compensation', linestyle='-', linewidth=0.5)
            self.line2, = self.axle.plot(self.time, self.actual_torque, 
                color='red', label="Actual", linestyle='-', linewidth=0.5)
            self.line3, = self.axle.plot(self.time, self.rotor_angle, 
                color='lightblue', label="Rotor angle", linestyle='-', linewidth=0.5)
            self.line4, = self.axle.plot(self.time, self.speed, 
                color='orange', label="Speed", linestyle='-', linewidth=0.5)
            plt.ylabel("Amplitude (p.u.)")
            plt.xlabel("Time (s)")
            plt.title("Torque plot")
            plt.legend(loc='upper right')
            axes = plt.gca()
            #axes.set_ylim([-0.20, 0.20])  # -0.20, 0.20
            axes.set_ylim([-1.1, 1.1])  # -0.20, 0.20
            #axes.set_xlim([0, 2])

    def step(self, value):     
        # The aim is to compensate disturbances, so we want the signal to
        # stay steady. Give positive reward if the value stays ~constant.
        # If the difference is huge, then give negative reward.

        # def getReward(reference, actual):
        #     #difference = abs(reference - actual)
        #     difference = abs(reference - actual)
        #     #reward = -(difference**2) # Use negative cost as reward:
        #     #reward = -0.2 * abs(actual - self.last_compensation_value)
        #     reward = -difference
        #     return reward

        def getReward(reference, actual):
            self.prev_actual = actual
            difference = abs(reference - actual)
            #out = difference + self.integral
            #self.integral = 0.2 * self.integral + difference
            #if self.integral > 20.0:
            #    self.integral = 20
            #print("integral: ", self.integral)          
            #cost = 2.0 * abs(actual - self.prev_actual) + abs(actual - reference)
            cost = difference
            return -cost

        # Done if some of the following conditions have occured:
        # 1) Max step number has been reached
        # 2) Fault has occured. Cannot continue
        # 3) Problem is solved if one full revolution with positive reward only
        def getDone():
            #done = self.step_num+1 >= self.max_steps \
            #    or self.sim.signal.getTripCode() != -1
            done = self.hasPeriodFinished()
            #        or self.sim.signal.getTripCode() != -1 \
            #        or self.step_num+1 >= self.max_steps
            # if self.sim.signal.getTripCode() != -1:
            #     self.hardReset()
            #done = self.step_num+1 >= self.max_steps
            return done

        new_value = 0
        if self.compensation:
            #new_value = self.last_compensation_value + value # roll attempt (continuous signal)
            #self.sim.signal.setCompensationTorque(float(new_value))
            self.sim.signal.setAction(value)
            self.last_compensation_value = value


        self.sim.command.step(self.step_size)
        #response_signal = self.sim.signal.gt_FA2TC_data_().n_meas
        #response_signal = self.sim.signal.getMeasuredSpeed()
        reward = getReward(0, self.sim.signal.getSimActualTorque())
        #reward = getReward(0.03, response_signal)
        #reward = getReward(60, response_signal)
        #reward = getReward(self.sim.signal.getSpeedReference(), response_signal)
        done = getDone()
        info = self.sim.signal.getTripCode()
        #info = self.last_compensation_value
        torque = self.sim.signal.getSimActualTorqueFiltered()
        state = np.array([torque, self.sim.signal.getRotorElectricalAngleCtrl(), new_value])
        self.step_num += 1

        return state, reward, done, info

    # Fast training: do not actually reset the whole sim
    # New period has started. Just reset steps.
    def reset(self):
        #print("step num:", self.step_num)
        self.integral = 0
        self.step_num = 0
        signal = self.sim.signal.getSimActualTorqueFiltered()
        angle = self.sim.signal.getRotorElectricalAngleCtrl()
        #print("angle:", angle) # Should be close to zero
        self.clearPlot()
        self.last_compensation_value = 0 # cannot stay the same as the end of the period. This is incorrect too...
        return np.array([signal, angle, 0])

        # fastReset = normal reset ^

        # HardReset -- use when trips. Restarts simulator
    def hardReset(self):
        del self.sim
        self.clearPlot()
        self.sim = Ilmarinen.SandboxApi()

        # Run to stable and then beginning of a electrical period
        self.sim.command.step(30)
        while not self.hasPeriodFinished():
            self.sim.command.step(0.001)

        return self.reset()

    def render(self):
        self.compensation_torque[self.step_num] = self.sim.signal.getAction()
        self.actual_torque[self.step_num] = self.sim.signal.getSimActualTorque()
        self.rotor_angle[self.step_num] = self.sim.signal.getRotorElectricalAngleCtrl()
        self.speed[self.step_num] = self.sim.signal.gt_FA2TC_data_().n_meas
        
        self.line1.set_ydata(self.compensation_torque)
        self.line2.set_ydata(self.actual_torque)
        self.line3.set_ydata(self.rotor_angle)
        self.line4.set_ydata(self.speed)

        self.figure.canvas.draw()
        self.figure.canvas.flush_events()
