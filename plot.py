import matplotlib.pyplot as plt
from matplotlib import cm
from matplotlib.colors import LightSource
import numpy as np

from Noise.perlin import terrain as PerlinTerrain
from Noise.value import terrain as ValueTerrain
from geoData.GMTED import everest_normalized, denali_normalized, matterhorn_normalized


terrain = PerlinTerrain
N = terrain.shape[0]
plot_type = int(input("(2)D or (3)D plot: "))
if plot_type == 2:
    # 2D plot
    plot = plt.imshow(terrain, interpolation=None, cmap=cm.gist_earth)
    plt.colorbar()
elif plot_type == 3:
    # 3D plot
    x = np.linspace(0, N, N)
    y = np.linspace(0, N, N)
    x, y = np.meshgrid(x, y)
    fig, ax = plt.subplots(subplot_kw=dict(projection='3d'))
    ls = LightSource(270, 45)
    rgb = ls.shade(terrain, cmap=cm.gist_earth, vert_exag=0.01, blend_mode="soft")
    surf = ax.plot_surface(x, y, terrain, rstride=1, cstride=1, facecolors=rgb,
                        linewidth=0, antialiased=False, shade=False)
plt.show()

"""fig, (ax1, ax2, ax3) = plt.subplots(1, 3, figsize=(18, 6))

# Plot Signal 1
im1 = ax1.imshow(everest_normalized, cmap=cm.gist_earth)
ax1.set_title('Everest')

# Plot Signal 2
im2 = ax2.imshow(denali_normalized, cmap=cm.gist_earth)
ax2.set_title('Denali')

# Plot Signal 3
im3 = ax3.imshow(matterhorn_normalized, cmap=cm.gist_earth)
ax3.set_title('Matterhorn')

# Adjust subplots to make room for the color bar
plt.subplots_adjust(right=0.85)  # Leave 15% space on the right

# Add a single color bar on the right side of the plots
cbar_ax = fig.add_axes([0.88, 0.15, 0.03, 0.7])  # [left, bottom, width, height]
fig.colorbar(im2, cax=cbar_ax)

plt.show()"""