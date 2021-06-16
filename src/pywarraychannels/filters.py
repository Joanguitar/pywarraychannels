import numpy as np

def __RCF_element__(rolloff_rc, t, M_rc):
    denominator = 1 - np.power((2 * rolloff_rc * t / M_rc), 2)
    if np.abs(denominator) < 1e-6:
        return np.sinc(1/(2*rolloff_rc))*(np.pi/4)
    else:
        return np.sinc(t / M_rc) * \
            np.cos(np.pi * rolloff_rc * t / M_rc) / \
            denominator

__RCF__ = np.vectorize(__RCF_element__)

### Filter classes
class RCFilter():
    def __init__(self, rolloff_rc = 0, M_rc = 1, early_samples = 0, padding = 0):
        self.rolloff_rc = rolloff_rc
        self.M_rc = M_rc
        self.early_samples = early_samples
        self.padding = padding
    def response(self, T, delay):
        tt = np.arange(T + 2*self.padding)
        if not np.isscalar(delay) and not len(np.array(delay).shape) == 0:
            tt = tt[:, np.newaxis]
            delay = np.array(delay)[np.newaxis, :]
        return __RCF__(self.rolloff_rc, (tt + self.early_samples + self.padding - delay), self.M_rc)
