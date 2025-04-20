# MORAN'S I
from scipy import sparse
from scipy.spatial.distance import cdist
import numpy as np
from Noise.perlin import terrain as PerlinTerrain
from Noise.value import terrain as ValueTerrain
from geoData.GMTED import everest_normalized, denali_normalized, matterhorn_normalized

def Moran(matrix, radius=16, block_size=1500):
    """
    Memory-optimized Moran's I calculation with block processing.
    
    Args:
        matrix: 2D numpy array (terrain data)
        radius: Neighborhood cutoff in pixels
        block_size: Number of points to process at once
        
    Returns:
        Moran's I, Expected I
    """
    
    # Flatten matrix and get basic stats
    y = matrix.flatten()
    y_mean = np.mean(y)
    deviations = y - y_mean
    n = len(y)
    
    # Generate coordinates
    rows, cols = matrix.shape
    x_coords, y_coords = np.meshgrid(np.arange(cols), np.arange(rows))
    coords = np.column_stack((x_coords.ravel(), y_coords.ravel()))
    
    # Initialize sparse weight matrix in LIL format (efficient for construction)
    weights = sparse.lil_matrix((n, n))
    
    # Process in blocks to limit memory usage
    for i in range(0, n, block_size):
        i_end = min(i + block_size, n)
        
        # Calculate distances for this block to all points
        dist_block = cdist(coords[i:i_end], coords, 'euclidean')
        
        # Apply inverse distance weighting within radius
        mask = (dist_block > 0) & (dist_block <= radius)
        dist_block[~mask] = 0
        dist_block[mask] = 1 / dist_block[mask]
        
        # Assign to sparse matrix
        for local_idx in range(i_end - i):
            global_idx = i + local_idx
            neighbors = np.where(dist_block[local_idx] > 0)[0]
            weights[global_idx, neighbors] = dist_block[local_idx, neighbors]
    
    # Convert to CSR format for efficient operations
    weights = weights.tocsr()
    
    # Row-standardize weights
    row_sums = np.array(weights.sum(axis=1)).flatten()
    row_sums[row_sums == 0] = 1  # Avoid division by zero
    weights = weights.multiply(1 / row_sums[:, np.newaxis])
    
    # Calculate Moran's I
    s0 = weights.sum()
    weighted_deviations = weights @ deviations
    numerator = (deviations * weighted_deviations).sum()
    denominator = (deviations ** 2).sum()
    morans_i = (n / s0) * (numerator / denominator)

    return morans_i

    
# Compute Moran's I
Ip = Moran(PerlinTerrain)
Iv = Moran(ValueTerrain)

print(f"Moran's I: {Ip}")
print(f"Moran's I: {Iv}")