import pywarraychannels
import numpy as np
import matplotlib.pyplot as plt

# Declare an uncertainty for the RX antenna (random uniform tilt)
uncertainty_RX = pywarraychannels.uncertainties.UniformTiltPanRoll(u_pan = False, u_roll = False)

# Define antennas
antenna_RX = pywarraychannels.antennas.LinearAntenna(32, uncertainty = uncertainty_RX)
antenna_TX = pywarraychannels.antennas.RectangularAntenna((8, 4))

# Define the channel class
channel = pywarraychannels.channels.AWGN(
    pywarraychannels.channels.MIMO(
        pywarraychannels.channels.Geometric(
            antenna_RX,
            antenna_TX,
            filter=pywarraychannels.filters.RCFilter(early_samples = 16, late_samples = 128)
            ),
        pilot=None),
    noise=0)

# Read TestRays
with open("Demos/TestRays.txt") as f:
    rays = pywarraychannels.em.Geometric([[float(p) for p in ray.split()] for ray in f.read().split("\n")[:-1]])

# Build the channel
channel.set_corr(antenna_RX.codebook_corr())
channel.build(rays)

# Measure the MIMO channel
measure = channel.measure()
print(measure.shape)

# Plot measure
plt.figure(1)
plt.plot(np.linalg.norm(measure, ord = 2, axis = 1))
plt.title("RX measurement spectrum")
plt.figure(2)
plt.plot(np.linalg.norm(measure, ord = 2, axis = 0))
plt.title("Time measurement spectrum")
plt.show()
