import pywarraychannels
import numpy as np
import matplotlib.pyplot as plt

### Declare an uncertainty for the RX antenna (random uniform tilt)
uncertainty_RX = pywarraychannels.uncertainties.UniformTiltPanRoll(u_pan = False, u_roll = False)

### Define antennas
antenna_RX = pywarraychannels.antennas.LinearAntenna(32, uncertainty = uncertainty_RX)
antenna_TX = pywarraychannels.antennas.RectangularAntenna((8, 16))

### Define the channel class
channel = pywarraychannels.channels.AWGN(pywarraychannels.channels.Geometric(antenna_RX, antenna_TX), noise = 0)

### Read TestRays
with open("Demo/TestRays.txt") as f:
    rays = pywarraychannels.em.Geometric([[float(p) for p in ray.split()] for ray in f.read().split("\n")[:-1]])

### Build the channel
channel.build(rays)

### Measure the channel (with default codebooks)
measure = channel.measure()

### Measure the channel in sequential pairs instead of all combinations
antenna_TX.set_codebook(antenna_TX.codebook[:, :32])
measure_seq = channel.measure(mode = "Sequential")
