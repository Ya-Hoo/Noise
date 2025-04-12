import matplotlib.pyplot as plt
import math
from Perlin.perlin2D import terrain, N

# HISTOGRAM
heights = [terrain[y][x] for x in range(N) for y in range(N)]
n_bins = math.ceil(2* (len(heights) ** (1/3)))

# Create the histogram
plt.figure(figsize=(10, 6))
plt.hist(heights, bins=n_bins, edgecolor='black', alpha=0.7, color='cornflowerblue')

# Add labels and title
plt.xlabel('Elevation Value')
plt.ylabel('Frequency')

# Show the plot
plt.grid(axis='y', alpha=0.5)
plt.show()