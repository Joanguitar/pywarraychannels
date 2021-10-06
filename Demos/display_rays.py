import pywarraychannels

# Read TestRays
with open("Demos/TestRays.txt") as f:
    rays = pywarraychannels.em.Geometric(
        [[float(p) for p in ray.split()] for ray in f.read().split("\n")[:-1]])

# Display rays
print(rays)
