import numpy as np
from Perlin.perlin2D import terrain, N

# POWER SPECTRAL DENSITY
def compute_psd_2d(matrix):
    # Remove DC component
    zero_mean = matrix - np.mean(matrix)
    
    # Compute 2D FFT
    dft = np.fft.fft2(zero_mean)
    
    # Compute PSD (|F(k,l)|² / N²) and convert to dB
    psd = 10 * np.log10((np.abs(dft)**2) / (matrix.shape[0] * matrix.shape[1]) + 1e-20)
    
    # Center the spectrum
    psd_shifted = np.fft.fftshift(psd)
    
    return psd_shifted

def plot_psd(psd):
    plt.figure(figsize=(10, 8))
    plt.imshow(psd, cmap='jet', 
               vmin=np.max([-60, np.min(psd)]),  # Set reasonable dB floor
               vmax=np.max(psd))
    plt.colorbar(label='Power (dB)')
    plt.axis('off')
    plt.show()

matrix = terrain
size = 1024
# Compute and plot PSD
psd = compute_psd_2d(matrix)
#plot_psd(psd)

def analyze_psd(psd):
    # Convert back to linear scale
    linear_psd = 10**(psd/10)  
    
    # Create frequency axes (normalized 0 to 0.5)
    freq = np.fft.fftshift(np.fft.fftfreq(1024))  
    
    # Radial profile
    radius = np.sqrt(np.add.outer(freq**2, freq**2))
    radial_profile = [linear_psd[(radius >= r-0.01) & (radius < r+0.01)].mean() 
                     for r in np.linspace(0, 0.5, 50)]
    
    plt.figure(figsize=(10,4))
    plt.plot(np.linspace(0, 0.5, 50), radial_profile)
    plt.xlabel('Spatial Frequency (cycles/pixel)')
    plt.ylabel('Power')
    plt.title('Radial Power Distribution')
    plt.grid()
    
    # Directional analysis
    directions = []
    for angle in range(0, 180, 10):
        mask = np.zeros_like(psd)
        cv2.line(mask, (512,512), 
                (int(512+500*np.cos(np.radians(angle))), 
                 int(512+500*np.sin(np.radians(angle)))),
                1, thickness=3)
        directions.append(linear_psd[mask==1].mean())
    
    plt.figure(figsize=(10,4))
    plt.plot(range(0, 180, 10), directions)
    plt.xlabel('Direction (degrees)')
    plt.ylabel('Power')
    plt.title('Directional Power Distribution')
    plt.grid()
    
analyze_psd(psd)
plt.show()