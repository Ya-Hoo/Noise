import matplotlib.pyplot as plt
import math
from Noise.perlin import N

from Noise.perlin import terrain as PerlinTerrain
from Noise.value import terrain as ValueTerrain


# HISTOGRAM
n_bins = math.ceil(2* (N ** (2/3)))

# Create the histogram
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 6))

ax1.hist(PerlinTerrain.flatten(), bins=n_bins, density=True, edgecolor='black', alpha=0.2, label='Perlin')
ax2.hist(ValueTerrain.flatten(), bins=n_bins, density=True, edgecolor='black', alpha=0.2, label='Value')

# Add labels and title
ax1.set_xlabel('Normalized Elevation Value [0,1]')
ax1.set_ylabel('Probability Density')
ax2.set_xlabel('Normalized Elevation Value [0,1]')
ax2.set_ylabel('Probability Density')

# Show the plot
ax1.grid(axis='y', alpha=0.5)
ax1.legend()
ax2.grid(axis='y', alpha=0.5)
ax2.legend()
plt.show()