import numpy as np
import matplotlib.pyplot as plt
import matplotlib.colors
import random, math

"""
Assumptions:
- Terrain constructed on 1024x1024 map
- normalise height to [0,1]
"""

np.random.seed(30042603)  # initialise seed for random number
random.seed(696969)
    
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
RES = (2,2)

# custom color map
cvals = [-1, 1]

# sunset?
colors = [ 
    "#393e75", "#644b80", "#8e588a", "#b5698e", "#dc7a91", "#ed8990", "#fe978e", "#fea98e", "#febb8e", "#ffd99e"
]

# fantasy
colors = [ 
    "#7c8d4c", "#b5ba61", "#e5d9c2", "#b6e3db"
][::-1]

# blue
colors = [
    "#f2f8e9", "#c5e2bf", "#95cac4", "#68a1c6", "#3b68a7"
]

# green
colors = [
    "#fefed1", "#cae3a1", "#8fc281", "#5d9f5c", "#33653c"
]

# idk
colors = [
    "#4e7bb1", "#b9d8e7", "#fefec6", "#ebb16e", "#bd3529"
]
norm = plt.Normalize(-1,1)
tuples = list(zip(map(norm,cvals), colors))
cmap = matplotlib.colors.LinearSegmentedColormap.from_list("", colors=colors)

terrain = generate_perlin_noise_2d(SHAPE, RES)
plot = plt.imshow(terrain, interpolation=None, cmap="viridis")
plt.colorbar()
plt.show()