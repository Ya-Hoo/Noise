from matplotlib import cm
from matplotlib.colors import LightSource
import matplotlib.pyplot as plt
import numpy as np
import random, json


np.random.seed(30042603)  # initialise seed for random number
random.seed(30042603)
    
def interpolant(t):
    return -2 * t*t*t + 3*t*t

def Value(x: float, y: float) -> float:
    # Identify grid cell vertices
    x0, y0 = int(x), int(y)
    x1, y1 = x0 + 1, y0 + 1
            
    # Compute dot products
    n00 = gradients[(x0, y0)]
    n10 = gradients[(x1, y0)]
    n01 = gradients[(x0, y1)]
    n11 = gradients[(x1, y1)]

    # Compute interpolation weights
    u, v = interpolant(x - x0), interpolant(y - y0)

    # Bilinear interpolation
    nx0 = (1 - u) * n00 + u * n10
    nx1 = (1 - u) * n01 + u * n11
    value = (1 - v) * nx0 + v * nx1

    return value

# Load parameters
with open("parameters.json", "r") as f:
    d = json.load(f)
N = d["N"]
FREQUENCY = d["FREQUENCY"]
AMPLITUDE = d["AMPLITUDE"]
LACUNARITY = d["LACUNARITY"]
PERSISTENCE = d["PERSISTENCE"]
OCTAVES_CNT = d["OCTAVES_CNT"]

# Generate gradient vectors for each grid point
gradients = {}
for i in range(N + 1):
    for j in range(N + 1):
        gradients[(i, j)] = random.uniform(0, 1)

terrain = np.zeros(shape=(N, N))
# OCTAVES
for i in range(OCTAVES_CNT):  
    for y in range(N):
        for x in range(N):
            terrain[y][x] = AMPLITUDE * Value(x/N * FREQUENCY, y/N * FREQUENCY)  
    FREQUENCY *= LACUNARITY
    AMPLITUDE *= PERSISTENCE
    

# Set up plot
x = np.linspace(0, N, N)
y = np.linspace(0, N, N)
x, y = np.meshgrid(x, y)
fig, ax = plt.subplots(subplot_kw=dict(projection='3d'))
ls = LightSource(270, 45)
rgb = ls.shade(terrain, cmap=cm.gist_earth, vert_exag=0.01, blend_mode="soft")
surf = ax.plot_surface(x, y, terrain, rstride=1, cstride=1, facecolors=rgb,
                       linewidth=0, antialiased=False, shade=False)

plt.show()