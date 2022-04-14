import pywarraychannels
import numpy as np
import matplotlib.pyplot as plt

# Declare antennas
lin_antenna = pywarraychannels.antennas.LinearAntenna(8)
rect_antenna = pywarraychannels.antennas.RectangularAntenna((8, 16), z_positive=True)

# Set reduced codebooks and plot them
# Linear
lin_antenna.set_reduced_codebook(4)
theta = np.linspace(-np.pi, np.pi, 1024)
AR = lin_antenna.array_factor([[np.cos(th), np.sin(th), 0] for th in theta])
plt.figure(0)
plt.plot(np.power(np.abs(AR), 2))
plt.draw()
# Rectangular
rect_antenna.set_reduced_codebook((2, 4))
AR = rect_antenna.array_factor([[np.cos(th), 0, np.sin(th)] for th in theta])
plt.figure(2)
plt.plot(np.power(np.abs(AR), 2))
plt.draw()
plt.show()
