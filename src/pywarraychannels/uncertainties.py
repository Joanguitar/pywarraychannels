import numpy as np


class Static():
    def __init__(self, tilt=0, pan=0, roll=0):
        self.state = [tilt, pan, roll]
        tilt_cos = np.cos(tilt)
        tilt_sin = np.sin(tilt)
        self.transform = np.array(
            [[tilt_cos, -tilt_sin, 0], [tilt_sin, tilt_cos, 0], [0, 0, 1]])
        pan_cos = np.cos(pan)
        pan_sin = np.sin(pan)
        self.transform = np.dot(self.transform, np.array(
            [[pan_cos, 0, -pan_sin], [0, 1, 0], [pan_sin, 0, pan_cos]]))
        roll_cos = np.cos(roll)
        roll_sin = np.sin(roll)
        self.transform = np.dot(self.transform, np.array(
            [[1, 0, 0], [0, roll_cos, -roll_sin], [0, roll_sin, roll_cos]]))

    def update(self):
        pass

    def apply(self, dir):
        return dir

    def apply_inverse(self, dir):
        return dir

    def __str__(self):
        return "Static: {:.1f} [deg]\nPan: {:.1f} [deg]\nRoll: {:.1f} [deg]".format(*self.state*180/np.pi)


class UniformTiltPanRoll():
    def __init__(self, u_tilt=True, u_pan=True, u_roll=True):
        self.u_tilt = u_tilt
        self.u_pan = u_pan
        self.u_roll = u_roll
        self.state = np.zeros(3)
        self.update()

    def update(self):
        if not (self.u_tilt or self.u_pan or self.u_roll):
            return
        self.transform = np.eye(3)
        if self.u_tilt:
            tilt = np.random.uniform(-np.pi, np.pi)
            self.state[0] = tilt
            tilt_cos = np.cos(tilt)
            tilt_sin = np.sin(tilt)
            self.transform = np.array(
                [[tilt_cos, -tilt_sin, 0], [tilt_sin, tilt_cos, 0], [0, 0, 1]])
        else:
            self.transform = np.eye(3)
        if self.u_pan:
            pan = np.random.uniform(-np.pi/2, np.pi/2)
            self.state[1] = pan
            pan_cos = np.cos(pan)
            pan_sin = np.sin(pan)
            self.transform = np.dot(self.transform, np.array(
                [[pan_cos, 0, -pan_sin], [0, 1, 0], [pan_sin, 0, pan_cos]]))
        if self.u_roll:
            roll = np.random.uniform(-np.pi, np.pi)
            self.state[2] = roll
            roll_cos = np.cos(roll)
            roll_sin = np.sin(roll)
            self.transform = np.dot(self.transform, np.array(
                [[1, 0, 0], [0, roll_cos, -roll_sin], [0, roll_sin, roll_cos]]))

    def apply(self, dir):
        if not (self.u_tilt or self.u_pan or self.u_roll):
            return dir
        else:
            return np.dot(dir, self.transform.T)

    def apply_inverse(self, dir):
        if not (self.u_tilt or self.u_pan or self.u_roll):
            return dir
        else:
            return np.dot(dir, self.transform)

    def __str__(self):
        return "Tilt: {:.1f} [deg]\nPan: {:.1f} [deg]\nRoll: {:.1f} [deg]".format(*self.state*180/np.pi)
