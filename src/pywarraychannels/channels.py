import numpy as np
import numexpr as ne
import pywarraychannels.filters

### Channel classes
class Geometric():
    def __init__(self, antenna_RX, antenna_TX, K = 128, f_c = 60e9, B = 1.760e9, filter = pywarraychannels.filters.RCFilter()):
        self.antenna_RX = antenna_RX
        self.antenna_TX = antenna_TX
        self.f_c = f_c
        self.B = B
        self.f_k = np.linspace(f_c-B/2, f_c+B/2, K, endpoint = False)+B/(2*K)
        self.f_k_rel = self.f_k/f_c
        self.filter = filter
    def build(self, rays):
        channel = np.zeros([len(self.antenna_RX.antenna_elements), len(self.antenna_TX.antenna_elements), len(self.f_k_rel)], dtype = "complex128")
        for phase, tau, power, doa_az, doa_el, dod_az, dod_el in rays:
            phase, doa_az, doa_el, dod_az, dod_el = np.radians(phase), np.radians(doa_az), np.radians(doa_el), np.radians(dod_az), np.radians(dod_el)   # Transform to radians
            doa = np.array([np.cos(doa_el)*np.cos(doa_az), np.cos(doa_el)*np.sin(doa_az), np.sin(doa_el)])                                              # Create direction vectors
            dod = np.array([np.cos(dod_el)*np.cos(dod_az), np.cos(dod_el)*np.sin(dod_az), np.sin(dod_el)])                                              # Create direction vectors
            scalar_doa = self.antenna_RX.scalar_dir(doa)
            scalar_dod = self.antenna_TX.scalar_dir(dod)
            response_time = np.sqrt(len(self.f_k_rel))*self.filter.response(len(self.f_k_rel), tau*self.B)
            complex_gain = np.power(10, (power-30)/20)*np.exp(1j*phase)
            #channel += complex_gain*np.exp(1j*np.pi*(scalar_doa[:, np.newaxis, np.newaxis]-scalar_dod[np.newaxis, :, np.newaxis])*f_k_rel[np.newaxis, np.newaxis, :])*response_time[np.newaxis, np.newaxis, :]
            channel += ne.evaluate("cg*(cos(pi*(sdoa-sdod)*fk)+1j*sin(pi*(sdoa-sdod)*fk))*tr", global_dict = \
                {"cg": complex_gain, "pi": np.pi, "sdoa": scalar_doa[:, np.newaxis, np.newaxis], "sdod": scalar_dod[np.newaxis, :, np.newaxis], "fk": self.f_k_rel[np.newaxis, np.newaxis, :], "tr": response_time[np.newaxis, np.newaxis, :]}) # Equivalente optimized line
        self.channel = channel
        return channel
    def measure(self, mode = "Pairs"):
        if mode == "Pairs":
            c_tx = np.tensordot(self.antenna_TX.codebook, self.channel, axes = (0, 1))
            rx_c_tx = np.tensordot(np.conj(self.antenna_RX.codebook), c_tx, axes = (0, 1))
            return rx_c_tx
        elif mode == "Sequential":
            rxtx_c = np.tensordot(self.antenna_TX.codebook[np.newaxis, :, :]*np.conj(self.antenna_RX.codebook)[:, np.newaxis, :], self.channel, axes = ([0, 1], [0, 1]))
            return rxtx_c
        else:
            print("Measure mode {} not recognized".format(mode))
            raise
    def __str__(self):
        return "Geometric channel\nSize: "+"x".join([str(a) for a in np.shape(self.channel)])+"\nEntries: "+" ".join([str(a) for a in np.ndarray.flatten(self.channel)])

class AWGN():
    def __init__(self, channel_dependency, power = 1, noise = 1e-1):
        self.channel_dependency = channel_dependency
        self.amp = np.sqrt(power)
        self.sigma = np.sqrt(noise/2)
    def build(self, *args, **kwargs):
        return self.channel_dependency.build(*args, **kwargs)
    def measure(self, *args, **kwargs):
        meas = self.channel_dependency.measure(*args, **kwargs)
        noise = self.sigma*(np.random.randn(*meas.shape)+1j*np.random.randn(*meas.shape))
        return self.amp*meas + noise
    def __str__(self):
        self.channel_dependency.print()
