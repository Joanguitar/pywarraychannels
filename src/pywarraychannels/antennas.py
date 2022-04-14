import numpy as np
import numexpr as ne
import scipy.fft as sfft
import pywarraychannels.uncertainties

### Basic antenna class
class Antenna():
    def __init__(self, antenna_elements, uncertainty=pywarraychannels.uncertainties.Static(), z_positive=False):
        """Example: Antenna([[0, 0, 0], [1, 0, 0], [0, 1, 0], [1, 1, 0]])"""
        self.antenna_elements = np.array(antenna_elements)
        self.uncertainty = uncertainty
        self.set_codebook(sfft.fft(np.eye(len(antenna_elements)))/np.sqrt(len(antenna_elements)))
        self.z_positive = z_positive
    def scalar_dir(self, dir):
        dir = np.array(dir)
        if len(dir.shape) == 1:
            dir = dir/np.linalg.norm(dir)
        else:
            dir = dir/np.linalg.norm(dir, axis = -1)[..., np.newaxis]
        dir = self.uncertainty.apply_inverse(dir)
        return np.dot(dir, self.antenna_elements.T)
    def steering_vector(self, dir):
        sdir = self.scalar_dir(dir) * np.pi
        if self.z_positive:
            dir = np.array(dir)
            dir = self.uncertainty.apply_inverse(dir)
            if len(dir.shape) == 1:
                return ne.evaluate("cos(sdir)+1j*sin(sdir)")*(dir[2] > 0)
            else:
                return ne.evaluate("cos(sdir)+1j*sin(sdir)")*(dir[..., 2] > 0)[..., np.newaxis]
        else:
            return ne.evaluate("cos(sdir)+1j*sin(sdir)")
    def array_factor(self, dir):
        return np.dot(np.conj(self.steering_vector(dir)), self.codebook)
    def update_uncertainty(self):
        self.uncertainty.update()
    def set_codebook(self, codebook):
        codebook = codebook/np.linalg.norm(codebook, ord = 2, axis = 0)[np.newaxis, ...]
        self.codebook = codebook
    def codebook_corr(self):
        return np.dot(np.conj(self.codebook.T), self.codebook)

### Basic antenna classes
class LinearAntenna(Antenna):
    def __init__(self, N_antennas, dir=[1, 0, 0], *args, **kwargs):
        """Example: LinearAntenna(8)"""
        dir = np.array(dir)
        super(LinearAntenna, self, *args).__init__(np.arange(N_antennas)[:, np.newaxis]*dir[np.newaxis, :], *args, **kwargs)
        self.N_antennas = N_antennas
        self.codebook = sfft.fft(np.eye(N_antennas))/np.sqrt(N_antennas)
    def set_reduced_codebook(self, n, overlap = True):
        if overlap:
            width = 2*np.pi/n+np.pi/self.N_antennas
        else:
            width = 2*np.pi/n
        bp = np.sinc((width/(2*np.pi))*(np.arange(self.N_antennas)-(self.N_antennas-1)/2))
        bp /= np.linalg.norm(bp)
        angles = np.linspace(0, 2*np.pi, n, endpoint = False)
        self.set_codebook(bp[:, np.newaxis]*np.exp(1j*np.arange(self.N_antennas)[:, np.newaxis]*angles[np.newaxis, :]))

class RectangularAntenna(Antenna):
    def __init__(self, N_antennas, dir=([1, 0, 0], [0, 1, 0]), *args, **kwargs):
        """Example: RectangularAntenna((8, 8))"""
        dir = np.array(dir)
        grid_y, grid_x = np.meshgrid(np.arange(N_antennas[1]), np.arange(N_antennas[0]))
        grid_x, grid_y = np.ndarray.flatten(grid_x), np.ndarray.flatten(grid_y)
        grid = np.array([grid_x, grid_y]).T
        super(RectangularAntenna, self, *args).__init__(np.dot(grid, dir), *args, **kwargs)
        self.N_antennas = N_antennas
        self.set_pair_codebook(sfft.fft(np.eye(N_antennas[0])), sfft.fft(np.eye(N_antennas[1])))
    def set_pair_codebook(self, cdb1, cdb2):
        cdb1 = cdb1/np.linalg.norm(cdb1, ord = 2, axis = 0)[np.newaxis, ...]
        cdb2 = cdb2/np.linalg.norm(cdb2, ord = 2, axis = 0)[np.newaxis, ...]
        self.cdb1 = cdb1
        self.cdb2 = cdb2
        super(RectangularAntenna, self).set_codebook(np.kron(cdb1, cdb2))
    def set_codebook(self, codebook):
        self.cdb1 = None
        self.cdb2 = None
        super(RectangularAntenna, self).set_codebook(codebook)
    def set_reduced_codebook(self, n, overlap=True):
        if overlap:
            width1, width2 = 2*np.pi/n[0]+np.pi/self.N_antennas[0], 2*np.pi/n[1]+np.pi/self.N_antennas[1]
        else:
            width1, width2 = 2*np.pi/n[0], 2*np.pi/n[1]
        bp1 = np.sinc((width1/(2*np.pi))*(np.arange(self.N_antennas[0])-(self.N_antennas[0]-1)/2))
        bp2 = np.sinc((width2/(2*np.pi))*(np.arange(self.N_antennas[1])-(self.N_antennas[1]-1)/2))
        bp1 /= np.linalg.norm(bp1)
        bp2 /= np.linalg.norm(bp2)
        angles1 = np.linspace(0, 2*np.pi, n[0], endpoint = False)
        angles2 = np.linspace(0, 2*np.pi, n[1], endpoint = False)
        cdb1 = bp1[:, np.newaxis]*np.exp(1j*np.arange(self.N_antennas[0])[:, np.newaxis]*angles1[np.newaxis, :])
        cdb2 = bp2[:, np.newaxis]*np.exp(1j*np.arange(self.N_antennas[1])[:, np.newaxis]*angles2[np.newaxis, :])
        self.set_pair_codebook(cdb1, cdb2)
