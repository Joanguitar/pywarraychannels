import numpy as np

# Auxiliar
def polar2cartesian(az, el):
    return [np.cos(az)*np.cos(el), np.sin(az)*np.cos(el), np.sin(el)]

def cartesian2polar(v):
    return np.asin(v[2]), np.angle(v[0]+v[1]*1j)

def wrapangle(a):
    return np.mod(a+np.pi, 2*np.pi)-np.pi

# Rays classes
class Geometric():
    def __init__(
        self, ray_info_or_phase, tau=None, power=None, doa_az=None,
        doa_el=None, dod_az=None, dod_el=None, bool_flip_RXTX=False):
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
        if bool_flip_RXTX:
            self.flip_RXTX()
    def flip_RXTX(self):
        ray_info = np.copy(self.ray_info)
        self.ray_info[:, 3] = ray_info[:, 5]
        self.ray_info[:, 4] = ray_info[:, 6]
        self.ray_info[:, 5] = ray_info[:, 3]
        self.ray_info[:, 6] = ray_info[:, 4]
    def classify_rays(self):
        epsilon = 1e-3
        c = 3e8
        bool_LoS = False
        classification = []
        for ii, (_, Tau, _, DoA_az, DoA_el, DoD_az, DoD_el) in enumerate(self):
            DoA_az, DoA_el, DoD_az, DoD_el = [
                np.deg2rad(a) for a in [DoA_az, DoA_el, DoD_az, DoD_el]]
            if ii == 0:
                Tau_0, DoA_az_0, DoA_el_0, DoD_az_0, DoD_el_0 =\
                    Tau, DoA_az, DoA_el, DoD_az, DoD_el
                DoA_0 = polar2cartesian(DoA_az, DoA_el)
                DoD_0 = polar2cartesian(DoD_az, DoD_el)
                if (np.abs(DoA_el+DoD_el) < epsilon) and\
                    (np.abs(wrapangle(DoA_az-DoD_az-np.pi)) < epsilon):
                    classification.append("LoS")
                    bool_LoS = True
                else:
                    classification.append("pseudo-LoS")
            else:
                DoA = polar2cartesian(DoA_az, DoA_el)
                DoD = polar2cartesian(DoD_az, DoD_el)
                cosa = np.dot(DoA, DoA_0)
                cosd = np.dot(DoD, DoD_0)
                sina = np.sqrt(1-cosa**2)
                sind = np.sqrt(1-cosd**2)
                sinad = cosa*sind + sina*cosd
                Tau_est = Tau_0*(sina+sind)/sinad
                if np.abs(Tau_est-Tau)*c < epsilon:
                    if bool_LoS:
                        kind = "1order"
                    else:
                        kind = "pseudo-1order"
                else:
                    kind = "NLoS"
                if np.abs(DoA_el+DoD_el) < epsilon:
                    kind += " (Wall)"
                if (np.abs(DoA_el-DoD_el) < epsilon) and\
                    (np.abs(wrapangle(DoA_az-DoA_az_0)) < epsilon) and\
                    (np.abs(wrapangle(DoD_az-DoD_az_0)) < epsilon):
                    if DoA_el < 0:
                        kind += " (Floor)"
                    else:
                        kind += " (Ceiling)"
                classification.append(kind)
        return classification
    def __iter__(self):
        return iter(self.ray_info)
    def __str__(self):
        format_str = "{:>12s} {:>12s} {:>12s} {:>12s} {:>12s} {:>12s} {:>12s} {:>20s}\n"
        prntstr = "  "+format_str.format("Phase [deg]", "Time [s]", "Power [dBm]", "DoA_az [deg]", "DoA_el [deg]", "DoD_az [deg]", "DoD_el [deg]", "kind    ")
        for ray, kind in zip(self.ray_info, self.classify_rays()):
            ray_str = [form.format(p) for form, p in zip(["{:.2f}", "{:.3e}", "{:.2f}", "{:.2f}", "{:.2f}", "{:.2f}", "{:.2f}"], ray)]
            prntstr += format_str.format(*ray_str, kind)
        return prntstr
