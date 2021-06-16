import numpy as np

### Filter classes
class RCFilter():
    def __init__(self, rolloff_rc = 0, M_rc = 1, early_samples = 0):
        self.rolloff_rc = rolloff_rc
        self.M_rc = M_rc
        self.early_samples = early_samples
    def response(self, T, delay):
        tt = np.arange(T)
        if np.isscalar(delay) or len(np.array(delay).shape) == 0:
            return np.sinc((tt + self.early_samples - delay) / self.M_rc) * \
                np.cos(np.pi * self.rolloff_rc * (tt + self.early_samples - delay) / self.M_rc) / \
                (1 - np.power((2 * self.rolloff_rc * (tt + self.early_samples - delay) / self.M_rc), 2))
        else:
            tt = tt[:, np.newaxis]
            delay = np.array(delay)[np.newaxis, :]
            denominator = 1 - np.power((2 * self.rolloff_rc * (tt + self.early_samples - delay) / self.M_rc), 2)
            if np.abs(denominator) < 1e-6:
                return np.sinc(1/(2*self.rolloff_rc))*(np.pi/4)
            else:
                return np.sinc((tt + self.early_samples - delay) / self.M_rc) * \
                    np.cos(np.pi * self.rolloff_rc * (tt + self.early_samples - delay) / self.M_rc) / \
                    denominator
