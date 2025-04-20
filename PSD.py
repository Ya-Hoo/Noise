import numpy as np
import matplotlib.pyplot as plt
from Noise.perlin import terrain as PerlinTerrain
from Noise.value import terrain as ValueTerrain

# Signal 1: Random noise + horizontal stripes
x1 = PerlinTerrain

# Signal 2: Random noise + checkerboard pattern
x2 = ValueTerrain

# Compute 2D PSD for both signals
def compute_2d_psd(signal):
    X = np.fft.fft2(signal)  # 2D FFT
    Sxx = np.abs(X)**2  # Power spectrum
    Sxx = np.fft.fftshift(Sxx)  # Shift zero-frequency to center
    return Sxx

Sxx1 = compute_2d_psd(x1)
Sxx2 = compute_2d_psd(x2)

# Convert to logarithmic scale and find global min/max for consistent color scaling
log_Sxx1 = np.log10(Sxx1)
log_Sxx2 = np.log10(Sxx2)
vmin = min(log_Sxx1.min(), log_Sxx2.min())  # Global minimum
vmax = max(log_Sxx1.max(), log_Sxx2.max())  # Global maximum

# Create a figure with two subplots and a shared color bar
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 6))

# Plot Signal 1 PSD
im1 = ax1.imshow(log_Sxx1, cmap='jet', vmin=vmin, vmax=vmax)
ax1.set_title('PSD of Perlin noise')
ax1.set_xlabel('Spatial Frequency (u)')
ax1.set_ylabel('Spatial Frequency (v)')

# Plot Signal 2 PSD
im2 = ax2.imshow(log_Sxx2, cmap='jet', vmin=vmin, vmax=vmax)
ax2.set_title('PSD of Value noise')
ax2.set_xlabel('Spatial Frequency (u)')
ax2.set_ylabel('Spatial Frequency (v)')

# Adjust subplots to make room for the color bar
plt.subplots_adjust(right=0.85)  # Leave 15% space on the right

# Add a single color bar on the right side of the plots
cbar_ax = fig.add_axes([0.88, 0.15, 0.03, 0.7])  # [left, bottom, width, height]
fig.colorbar(im2, cax=cbar_ax, label='Log Power (dB)')

plt.show()