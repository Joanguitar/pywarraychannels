import pywarraychannels
import numpy as np
import matplotlib.pyplot as plt

# Declare antennas
lin_antenna = pywarraychannels.antennas.LinearAntenna(8)
rect_antenna = pywarraychannels.antennas.RectangularAntenna((8, 16))

# Direction the beam-pattern is pointing to
dir = np.asarray([0, 0, 1])
dir = dir / np.linalg.norm(dir, axis=-1)[..., np.newaxis]

# Angular domain
theta = np.linspace(-np.pi, np.pi, 1024)

# Set reduced codebooks and plot them
# Linear
steering_vector = lin_antenna.steering_vector(dir)
lin_antenna.set_codebook(steering_vector)
AR = lin_antenna.array_factor([[np.cos(th), np.sin(th), 0] for th in theta])
plt.figure(0)
plt.plot(np.cos(theta)*np.power(np.abs(AR), 2), np.sin(theta)*np.power(np.abs(AR), 2))
plt.xlim([-8, 8])
plt.ylim([-8, 8])
plt.xlabel("x-axis")
plt.ylabel("y-axis")
# Rectangular
steering_vector = rect_antenna.steering_vector(dir)
rect_antenna.set_codebook(steering_vector)
AR = rect_antenna.array_factor([[np.cos(th), 0, np.sin(th)] for th in theta])
plt.figure(1)
plt.plot(np.cos(theta)*np.power(np.abs(AR), 2), np.sin(theta)*np.power(np.abs(AR), 2))
plt.xlim([-128, 128])
plt.ylim([-128, 128])
plt.xlabel("x-axis")
plt.ylabel("z-axis")
plt.show()
