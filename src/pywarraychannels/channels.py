import numpy as np
import numexpr as ne
import pywarraychannels.filters
import scipy.ndimage as sndimage

### Channel classes
class Geometric():
    def __init__(self, antenna_RX, antenna_TX, K=128, f_c=60e9, B=1.760e9, filter=pywarraychannels.filters.RCFilter(), bool_sync=True):
        self.antenna_RX = antenna_RX
        self.antenna_TX = antenna_TX
        self.f_c = f_c
        self.B = B
        self.f_k = np.linspace(f_c-B/2, f_c+B/2, K, endpoint=False)+B/(2*K)
        self.f_k_rel = self.f_k/f_c
        self.filter = filter
        self.bool_sync = bool_sync
    def build(self, rays):
        channel = None
        if len([ray for ray in rays]) == 0:
            response_time = self.filter.response(len(self.f_k_rel), 0)
            scalar_doa = self.antenna_RX.scalar_dir([0, 0, 0])
            scalar_dod = self.antenna_TX.scalar_dir([0, 0, 0])
            self.channel = np.zeros((len(scalar_doa), len(scalar_dod), len(response_time)))
            return self.channel.copy()
        if self.bool_sync:
            tau_min = np.min([ray[1] for ray in rays])
        else:
            tau_min = 0
        for phase, tau, power, doa_az, doa_el, dod_az, dod_el in rays:
            phase, doa_az, doa_el, dod_az, dod_el = np.radians(phase), np.radians(doa_az), np.radians(doa_el), np.radians(dod_az), np.radians(dod_el)   # Transform to radians
            doa = np.array([np.cos(doa_el)*np.cos(doa_az), np.cos(doa_el)*np.sin(doa_az), np.sin(doa_el)])                                              # Create direction vectors
            dod = np.array([np.cos(dod_el)*np.cos(dod_az), np.cos(dod_el)*np.sin(dod_az), np.sin(dod_el)])                                              # Create direction vectors
            scalar_doa = self.antenna_RX.scalar_dir(doa)
            scalar_dod = self.antenna_TX.scalar_dir(dod)
            response_time = self.filter.response(len(self.f_k_rel), (tau-tau_min)*self.B+(scalar_doa[:, np.newaxis]-scalar_dod[np.newaxis, :])*self.B/self.f_c)
            complex_gain = np.power(10, (power-30)/20)*np.exp(1j*phase)
            #channel += complex_gain*np.exp(1j*np.pi*(scalar_doa[:, np.newaxis, np.newaxis]-scalar_dod[np.newaxis, :, np.newaxis])*f_k_rel[np.newaxis, np.newaxis, :])*response_time[np.newaxis, np.newaxis, :]
            if channel is None:
                channel = ne.evaluate("cg*(cos(pi*(sdoa-sdod))+1j*sin(pi*(sdoa-sdod)))*tr", global_dict = \
                    {"cg": complex_gain, "pi": np.pi, "sdoa": scalar_doa[:, np.newaxis, np.newaxis], "sdod": scalar_dod[np.newaxis, :, np.newaxis], "tr": np.transpose(response_time, [1, 2, 0])}) # Equivalente optimized line
            else:
                channel += ne.evaluate("cg*(cos(pi*(sdoa-sdod))+1j*sin(pi*(sdoa-sdod)))*tr", global_dict = \
                    {"cg": complex_gain, "pi": np.pi, "sdoa": scalar_doa[:, np.newaxis, np.newaxis], "sdod": scalar_dod[np.newaxis, :, np.newaxis], "tr": np.transpose(response_time, [1, 2, 0])}) # Equivalente optimized line
        self.channel = channel
        return channel
    def measure(self, signal=None, mode="Pairs"):
        if signal is None:
            signal = [np.sqrt(len(self.f_k_rel))]
        if mode == "Pairs":
            c_tx = np.tensordot(self.antenna_TX.codebook, self.channel, axes = (0, 1))
            rx_c_tx = np.tensordot(np.conj(self.antenna_RX.codebook), c_tx, axes = (0, 1))
            return sndimage.convolve1d(np.pad(rx_c_tx, ((0, 0), (0, 0), (len(signal)-1, 0))), signal, axis = 2)
        elif mode == "Sequential":
            rxtx_c = np.tensordot(self.antenna_TX.codebook[np.newaxis, :, :]*np.conj(self.antenna_RX.codebook)[:, np.newaxis, :], self.channel, axes = ([0, 1], [0, 1]))
            return sndimage.convolve1d(np.pad(rxtx_c, ((0, 0), (len(signal)-1, 0))), signal, axis = 1)
        else:
            print("Measure mode {} not recognized".format(mode))
            raise
    def __str__(self):
        return "Geometric channel\nSize: "+"x".join([str(a) for a in np.shape(self.channel)])+"\nEntries: "+" ".join([str(a) for a in np.ndarray.flatten(self.channel)])

class MIMO():
    def __init__(self, channel_dependency, pilot=None):
        self.channel_dependency = channel_dependency
        self.pilot = pilot
    def build(self, *args, **kwargs):
        return self.channel_dependency.build(*args, **kwargs)
    def measure(self, *args, **kwargs):
        meas_tap = self.channel_dependency.measure(*args, **kwargs)
        M_RX, M_TX, D = meas_tap.shape
        pilot = self.pilot or np.concatenate(
            [
                np.zeros((M_TX, D)),
                np.fft.fft(np.eye(M_TX)),
                np.zeros((M_TX, D-M_TX))
            ], axis=1)
        meas = np.zeros((M_RX, pilot.shape[1]-D), dtype="complex")
        for d in range(D):
            meas += np.dot(meas_tap[:, :, d], pilot[:, D-d-1:-d-1])
        return meas
    def __str__(self):
        return "MIMO + "+str(self.channel_dependency)

class AWGN():
    def __init__(self, channel_dependency, power=1, noise=1e-1, corr=None):
        self.channel_dependency = channel_dependency
        self.amp = np.sqrt(power)
        self.sigma = np.sqrt(noise/2)
    def build(self, *args, **kwargs):
        return self.channel_dependency.build(*args, **kwargs)
    def measure(self, *args, **kwargs):
        meas = self.channel_dependency.measure(*args, **kwargs)
        noise = self.sigma*(np.random.randn(*meas.shape)+1j*np.random.randn(*meas.shape))
        if self.L is None:
            return self.amp*meas + noise
        else:
            return self.amp*meas + np.dot(self.L, noise)
    def set_corr(self, corr=None):
        if corr is None:
            self.L = None
        else:
            self.L = np.linalg.cholesky(corr)
    def __str__(self):
        return "AWGN + "+str(self.channel_dependency)

class Rician():
    def __init__(self, channel_dependency, k=1):
        self.channel_dependency = channel_dependency
        self.k = k
    def build(self, *args, **kwargs):
        main_channel =  self.channel_dependency.build(*args, **kwargs)
        rician_component = np.random.randn(*main_channel.shape)
        rician_component *= np.linalg.norm(main_channel)*np.sqrt(self.k)/np.linalg.norm(rician_component)
        self.rician_component = rician_component
        return main_channel + rician_component
    def measure(self, signal=None, mode="Pairs", *args, **kwargs):
        meas = self.channel_dependency.measure(signal=signal, mode=mode, *args, **kwargs)
        if signal is None:
            signal = [np.sqrt(len(self.channel_dependency.f_k_rel))]
        if mode == "Pairs":
            c_tx = np.tensordot(self.channel_dependency.antenna_TX.codebook, self.rician_component, axes = (0, 1))
            rx_c_tx = np.tensordot(np.conj(self.channel_dependency.antenna_RX.codebook), c_tx, axes = (0, 1))
            return meas + sndimage.convolve1d(np.pad(rx_c_tx, ((0, 0), (0, 0), (len(signal)-1, 0))), signal, axis = 2)
        elif mode == "Sequential":
            rxtx_c = np.tensordot(self.channel_dependency.antenna_TX.codebook[np.newaxis, :, :]*np.conj(self.channel_dependency.antenna_RX.codebook)[:, np.newaxis, :], self.rician_component, axes = ([0, 1], [0, 1]))
            return meas + sndimage.convolve1d(np.pad(rxtx_c, ((0, 0), (len(signal)-1, 0))), signal, axis = 1)
        else:
            print("Measure mode {} not recognized".format(mode))
            raise
    def __str__(self):
        return "Rician + "+str(self.channel_dependency)
