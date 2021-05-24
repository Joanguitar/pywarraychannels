import numpy as np

### Rays classes
class Geometric():
    def __init__(self, ray_info_or_phase, tau = None, power = None, doa_az = None, doa_el = None, dod_az = None, dod_el = None):
        if tau is None:
            ray_info = np.array(ray_info_or_phase)
            if len(ray_info.shape) == 1:
                ray_info = ray_info[np.newaxis, :]
        else:
            ray_info = np.array([ray_info_or_phase, tau, power, doa_az, doa_el, dod_az, dod_el])
            if len(ray_info.shape) == 1:
                ray_info = ray_info[np.newaxis, :]
            else:
                ray_info = ray_info[np.newaxis, :].T
        self.ray_info = ray_info
    def __iter__(self):
        return iter(self.ray_info)
    def __str__(self):
        format_str = "{:>12s} {:>12s} {:>12s} {:>12s} {:>12s} {:>12s} {:>12s}\n"
        prntstr = "  "+format_str.format("Phase [deg]", "Time [s]", "Power [dBm]", "DoA_az [deg]", "DoA_el [deg]", "DoD_az [deg]", "DoD_el [deg]")
        for ray in self.ray_info:
            ray_str = [form.format(p) for form, p in zip(["{:.2f}", "{:.3e}", "{:.2f}", "{:.2f}", "{:.2f}", "{:.2f}", "{:.2f}"], ray)]
            prntstr += format_str.format(*ray_str)
        return prntstr
