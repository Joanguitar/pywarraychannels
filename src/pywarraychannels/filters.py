import numpy as np

def __RCF__(rolloff_rc, t, M_rc):
    denominator = 1 - np.power((2 * rolloff_rc * t / M_rc), 2)
    if rolloff_rc == 0:
        inder_val = 0
    else:
        inder_val = np.sinc(1/(2*rolloff_rc))*(np.pi/4)
    sinc_val = np.sinc(t / M_rc)
    cos_val =np.cos(np.pi * rolloff_rc * t / M_rc)
    denominator_no0 = denominator
    denominator_no0[np.abs(denominator) < 1e-6] = 1
    return np.where(np.abs(denominator) < 1e-6, inder_val, sinc_val * cos_val / denominator_no0)

### Filter classes
class RCFilter():
    def __init__(self, rolloff_rc = 0, M_rc = 1, early_samples = 0, late_samples = 0):
        self.rolloff_rc = rolloff_rc
        self.M_rc = M_rc
        self.early_samples = early_samples
        self.late_samples = late_samples
    def response(self, T, delay):
        tt = np.arange(T + self.early_samples + self.late_samples)
        if not np.isscalar(delay) and not len(np.array(delay).shape) == 0:
            delay = np.array(delay)[np.newaxis, ...]
            tt = np.expand_dims(tt, tuple(np.arange(1, len(delay.shape))))
        return __RCF__(self.rolloff_rc, (tt - self.early_samples - delay), self.M_rc)
