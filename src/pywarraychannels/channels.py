import numpy as np
import numexpr as ne
import pywarraychannels.filters

### Channel class
class AWGN():
    def __init__(self, antenna_RX, antenna_TX, K = 128, f_c = 60e9, B = 1.760e9, N = 1e-20, filter = pywarraychannels.filters.RCFilter()):
        self.antenna_RX = antenna_RX
        self.antenna_TX = antenna_TX
        self.f_c = f_c
        self.B = B
        self.sigma = np.sqrt(N)
        self.f_k = np.linspace(f_c-B/2, f_c+B/2, K, endpoint = False)+B/(2*K)
        self.f_k_rel = self.f_k/f_c
        self.filter = filter
    def build(self, info):
        channel = np.zeros([len(self.antenna_RX.antenna_elements), len(self.antenna_TX.antenna_elements), len(self.f_k_rel)], dtype = "complex128")
        for phase, tau, power, doa_az, doa_el, dod_az, dod_el in info:
            doa_az, doa_el, dod_az, dod_el = np.radians(doa_az), np.radians(doa_el), np.radians(dod_az), np.radians(dod_el) # Transform to radians
            doa = np.array([np.cos(doa_el)*np.cos(doa_az), np.cos(doa_el)*np.sin(doa_az), np.sin(doa_el)])                  # Create direction vectors
            dod = np.array([np.cos(dod_el)*np.cos(dod_az), np.cos(dod_el)*np.sin(dod_az), np.sin(dod_el)])                  # Create direction vectors
            scalar_doa = self.antenna_RX.scalar_dir(doa)
            scalar_dod = self.antenna_TX.scalar_dir(dod)
            response_time = self.filter.response(len(self.f_k_rel), tau*self.B)
            complex_gain = np.power(10, (power-30)/20)*np.exp(1j*phase)
            #channel += complex_gain*np.exp(1j*np.pi*(scalar_doa[:, np.newaxis, np.newaxis]-scalar_dod[np.newaxis, :, np.newaxis])*f_k_rel[np.newaxis, np.newaxis, :])*response_time[np.newaxis, np.newaxis, :]
            channel += ne.evaluate("cg*(cos(pi*(sdoa-sdod)*fk)+1j*sin(pi*(sdoa-sdod)*fk))*tr", global_dict = \
                {"cg": complex_gain, "pi": np.pi, "sdoa": scalar_doa[:, np.newaxis, np.newaxis], "sdod": scalar_dod[np.newaxis, :, np.newaxis], "fk": self.f_k_rel[np.newaxis, np.newaxis, :], "tr": response_time[np.newaxis, np.newaxis, :]}) # Equivalente optimized line
        self.channel = channel
        return channel
    def measure(self):
        c_tx = np.tensordot(self.antenna_TX.codebook, self.channel, axes = (1, 1))
        rx_c_tx = np.tensordot(np.conj(self.antenna_RX.codebook), c_tx, axes = (1, 1))
        noise = (self.sigma/np.sqrt(2))*(np.random.randn(*rx_c_tx.shape)+1j*np.random.randn(*rx_c_tx.shape))
        return rx_c_tx + noise
    def print(self):
        print("Size: "+"x".join([str(a) for a in np.shape(self.channel)]))
        print("Entries: "+" ".join([str(a) for a in np.ndarray.flatten(self.channel)]))
