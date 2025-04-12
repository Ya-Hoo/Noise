import numpy as np
import matplotlib.pyplot as plt
from matplotlib import cm
import random, math, json

np.random.seed(30042603)  # initialise seed for random number
random.seed(30042603)
    
def interpolant(t):
    return -2 * t*t*t + 3*t*t

def dotProduct(ix, iy, x, y, gradientVec):
    dx, dy = x - ix, y - iy
    return dx * gradientVec[0] + dy * gradientVec[1]

def Perlin(x: float, y: float) -> float:
    # Identify grid cell vertices
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
        angle = random.uniform(0, 2 * math.pi)
        gradients[(i, j)] = (math.cos(angle), math.sin(angle))

terrain = np.zeros(shape=(N, N))

# OCTAVES
for i in range(OCTAVES_CNT):  
    for y in range(N):
        for x in range(N):
            terrain[y][x] = Perlin(x/N * FREQUENCY, y/N * FREQUENCY)  
    FREQUENCY *= LACUNARITY
    AMPLITUDE *= PERSISTENCE

# PLOTTING
plot = plt.imshow(terrain, interpolation=None, cmap=cm.gist_earth)
plt.colorbar()
plt.show()

"""# HISTOGRAM
heights = [terrain[y][x] for x in range(N) for y in range(N)]
n_bins = math.ceil(2* (len(heights) ** (1/3)))
print(n_bins)
# Create the histogram
plt.figure(figsize=(10, 6))
plt.hist(heights, bins=n_bins, edgecolor='black', alpha=0.7, color='cornflowerblue')

# Add labels and title
plt.xlabel('Elevation Value')
plt.ylabel('Frequency')

# Show the plot
plt.grid(axis='y', alpha=0.5)
plt.show()"""

"""# MORAN'S I
from scipy import sparse
from scipy.spatial.distance import cdist
from scipy.stats import norm

def calculate_morans_i_optimized(matrix, radius=7, block_size=500):
    matrix = np.asarray(matrix)
    rows, cols = matrix.shape
    n = rows * cols
    y = matrix.flatten()
    y_mean = np.mean(y)
    deviations = y - y_mean
    
    # Create coordinate grid
    x_coords, y_coords = np.meshgrid(np.arange(cols), np.arange(rows))
    coords = np.column_stack((x_coords.flatten(), y_coords.flatten()))
    
    # Initialize sparse weight matrix
    weights = sparse.lil_matrix((n, n))
    
    # Process lower triangle only (exploit symmetry)
    for i in range(0, n, block_size):
        i_end = min(i + block_size, n)
        dist_block = cdist(coords[i:i_end], coords[i:], 'euclidean')
        
        # Apply inverse distance weighting within radius
        mask = (dist_block > 0) & (dist_block <= radius)
        dist_block[~mask] = 0
        dist_block[mask] = 1 / dist_block[mask]
        
        # Assign symmetric weights
        for block_row in range(i_end - i):
            global_row = i + block_row
            for block_col in range(block_row, dist_block.shape[1]):
                global_col = i + block_col
                val = dist_block[block_row, block_col]
                if val != 0:
                    weights[global_row, global_col] = val
                    weights[global_col, global_row] = val
    
    # Convert to CSR format for efficient operations
    weights = weights.tocsr()
    
    # Row standardization
    row_sums = np.array(weights.sum(axis=1)).flatten()
    row_sums[row_sums == 0] = 1  # avoid division by zero
    weights = weights.multiply(1 / row_sums[:, np.newaxis])
    
    # Calculate Moran's I
    s0 = weights.sum()
    weighted_deviations = weights @ deviations
    numerator = (deviations * weighted_deviations).sum()
    denominator = (deviations ** 2).sum()
    morans_i = (n / s0) * (numerator / denominator)
    
    # Expected value under randomness
    expected_i = -1 / (n - 1)
    
    # Calculate variance (using symmetric weights)
    W_sym = (weights + weights.T)/2
    s1 = 0.5 * (W_sym.power(2).sum() + W_sym.diagonal().sum())
    s2 = 4 * (W_sym.sum(axis=0) ** 2).sum()
    
    n_sq = n * n
    s0_sq = s0 * s0
    term1 = (n_sq - 3*n + 3)*s1 - n*s2 + 3*s0_sq
    term2 = (n_sq - n)*s1 - 2*n*s2 + 6*s0_sq
    denominator_var = (n - 1)*(n - 2)*(n - 3)*s0_sq
    variance_i = (term1 / denominator_var) - (term2 / denominator_var) - (expected_i ** 2)
    
    # Z-score and p-value
    z_score = (morans_i - expected_i) / np.sqrt(variance_i)
    p_value = 2 * (1 - norm.cdf(abs(z_score)))
    
    return morans_i, expected_i, z_score, p_value

print(calculate_morans_i_optimized(terrain))"""

""" PSD
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

# Example usage with a simple 16x16 matrix
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
plt.show()"""