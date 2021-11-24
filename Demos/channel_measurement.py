import pywarraychannels
import numpy as np
import matplotlib.pyplot as plt

# Declare an uncertainty for the RX antenna (random uniform tilt)
uncertainty_RX = pywarraychannels.uncertainties.UniformTiltPanRoll(u_pan = False, u_roll = False)

# Define antennas
antenna_RX = pywarraychannels.antennas.LinearAntenna(32, uncertainty = uncertainty_RX)
antenna_TX = pywarraychannels.antennas.RectangularAntenna((8, 16))

# Define the channel class
channel = pywarraychannels.channels.AWGN(
    pywarraychannels.channels.Rician(
        pywarraychannels.channels.Geometric(
            antenna_RX,
            antenna_TX,
            filter=pywarraychannels.filters.RCFilter(early_samples = 16, late_samples = 128)
            ),
        k=1),
    noise=0)

# Read TestRays
with open("Demos/TestRays.txt") as f:
    rays = pywarraychannels.em.Geometric([[float(p) for p in ray.split()] for ray in f.read().split("\n")[:-1]])

# Build the channel
channel.build(rays)

# Measure the channel (with default codebooks)
measure = channel.measure()

# Measure the channel in sequential pairs instead of all combinations
antenna_TX.set_codebook(antenna_TX.codebook[:, :32])
measure_seq = channel.measure(mode = "Sequential")

# Plot measure
plt.figure(1)
plt.plot(np.linalg.norm(measure, ord = "fro", axis = (1, 2)))
plt.title("RX measurement spectrum")
plt.figure(2)
plt.imshow(np.reshape(np.linalg.norm(measure, ord = "fro", axis = (0, 2)), (8, 16)))
plt.title("TX measurement spectrum")
plt.figure(3)
plt.plot(np.linalg.norm(np.reshape(measure, [-1, measure.shape[2]]), ord = 2, axis = 0))
plt.title("Time measurement spectrum")
plt.show()
