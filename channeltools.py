import numpy as np
import numexpr as ne
import scipy.fft as sfft

### Basic antenna class
class Antenna():
    def __init__(self, antenna_elements):
        """Example: Antenna([[0, 0, 0], [1, 0, 0], [0, 1, 0], [1, 1, 0]])"""
        self.antenna_elements = np.array(antenna_elements)
        self.codebook = sfft.fft(np.eye(len(antenna_elements)))/np.sqrt(len(antenna_elements))
    def scalar_dir(self, dir):
        dir = np.array(dir)
        dir = dir/np.linalg.norm(dir, axis = 0)
        return np.dot(dir, self.antenna_elements.T)
    def response(self, dir):
        sdir = self.scalar_dir(dir)
        return ne.evaluate("cos(sdir)+1i*sin(sdir)")

### Basic antenna classes
class LinearAntenna(Antenna):
    def __init__(self, N_antennas, dir = [1, 0, 0]):
        """Example: LinearAntenna(8)"""
        dir = np.array(dir)
        super(LinearAntenna, self).__init__(np.arange(N_antennas)[:, np.newaxis]*dir[np.newaxis, :])
        self.codebook = sfft.fft(np.eye(N_antennas))/np.sqrt(N_antennas)

class RectangularAntenna(Antenna):
    def __init__(self, N_antennas, dir = ([1, 0, 0], [0, 1, 0])):
        """Example: RectangularAntenna([8, 8])"""
        dir = np.array(dir)
        grid_x, grid_y = np.meshgrid(np.arange(N_antennas[0]), np.arange(N_antennas[1]))
        grid_x, grid_y = np.ndarray.flatten(grid_x), np.ndarray.flatten(grid_y)
        grid = np.array([grid_x, grid_y]).T
        super(RectangularAntenna, self).__init__(np.dot(grid, dir))
        self.codebook = np.kron(sfft.fft(np.eye(N_antennas[0])), sfft.fft(np.eye(N_antennas[1])))/np.sqrt(N_antennas[0]*N_antennas[1])

### Filter classes
class RCFilter():
    def __init__(self, rolloff_rc = 0, M_rc = 1, early_samples = 0):
        self.rolloff_rc = rolloff_rc
        self.M_rc = M_rc
        self.early_samples = early_samples
    def response(self, T, delay):
        tt = np.arange(T)
        return np.sinc((tt + self.early_samples - delay) / self.M_rc) * \
            np.cos(np.pi * self.rolloff_rc * (tt + self.early_samples - delay) / self.M_rc) / \
            (1 - np.power((2 * self.rolloff_rc * (tt + self.early_samples - delay) / self.M_rc), 2))

### Channel class
class Channel():
    def __init__(self, antenna_RX, antenna_TX, K = 128, f_c = 60e9, B = 1.760e9, N = 1e-20, filter = RCFilter()):
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
