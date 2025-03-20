from matplotlib import cbook
from matplotlib import cm
from matplotlib.colors import LightSource
import matplotlib.pyplot as plt
import numpy as np
import random, math

random.seed(30042603)
np.random.seed(30042603)

def interpolant(t):
    return 6 * t**5 - 15 * t**4 + 10 * t**3

def dotProduct(ix, iy, x, y, gradientVec):
    dx, dy = x - ix, y - iy
    return dx * gradientVec[0] + dy * gradientVec[1]

def generate_perlin_noise_2d(shape, res):
    width, height = shape
    grid_x, grid_y = res

    # Generate gradient vectors for each grid point
    gradients = {}
    for i in range(grid_x + 1):
        for j in range(grid_y + 1):
            angle = random.uniform(0, 2 * math.pi)
            gradients[(i, j)] = (math.cos(angle), math.sin(angle))
    #print(gradients)

    # Compute noise values for each pixel
    noise = [[0] * width for _ in range(height)]
    
    for i in range(height):
        for j in range(width):
            # Convert pixel coordinates to grid space
            x = (j / width) * grid_x
            y = (i / height) * grid_y
            
            # Identify grid cell corners
            x0, y0 = int(x), int(y)
            x1, y1 = x0 + 1, y0 + 1
            
            # Compute dot products
            n00 = dotProduct(x0, y0, x, y, gradients[(x0, y0)])
            n10 = dotProduct(x1, y0, x, y, gradients[(x1, y0)])
            n01 = dotProduct(x0, y1, x, y, gradients[(x0, y1)])
            n11 = dotProduct(x1, y1, x, y, gradients[(x1, y1)])

            # Compute interpolation weights
            u, v = interpolant(x - x0), interpolant(y - y0)

            # Bilinear interpolation
            nx0 = (1 - u) * n00 + u * n10
            nx1 = (1 - u) * n01 + u * n11
            value = (1 - v) * nx0 + v * nx1

            # Store result in noise array
            noise[i][j] = value

    return noise

SHAPE = (1024, 1024) # width * height
RES = (1,1)
FREQUENCY = 1
AMPLITUDE = 1
LACUNARITY = 2
PERSISTENCE = 0.5
OCTAVES_CNT = 6

terrain = np.zeros(shape=SHAPE)

# OCTAVES
for i in range(OCTAVES_CNT):    
    terrain += AMPLITUDE * np.array(generate_perlin_noise_2d(SHAPE, (RES[0]*FREQUENCY,RES[1]*FREQUENCY)))
    FREQUENCY *= LACUNARITY
    AMPLITUDE *= PERSISTENCE

nrows, ncols = SHAPE
x = np.linspace(0, SHAPE[0], ncols)
y = np.linspace(0, SHAPE[1], nrows)
x, y = np.meshgrid(x, y)

# Set up plot
fig, ax = plt.subplots(subplot_kw=dict(projection='3d'))
ls = LightSource(270, 45)
rgb = ls.shade(terrain, cmap=cm.gist_earth, vert_exag=0.25, blend_mode="soft")
surf = ax.plot_surface(x, y, terrain, rstride=1, cstride=1, facecolors=rgb,
                       linewidth=0, antialiased=False, shade=False)

plt.show()