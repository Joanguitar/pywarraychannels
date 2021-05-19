import numpy as np
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
