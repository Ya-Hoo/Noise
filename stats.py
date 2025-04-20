import numpy as np
from scipy import stats
from scipy.stats import wasserstein_distance
from Noise.perlin import terrain as PerlinTerrain
from Noise.value import terrain as ValueTerrain
from geoData.GMTED import everest_normalized, denali_normalized, matterhorn_normalized


def calculate_statistics(array_2d):
    """
    Calculate various statistics from a 2D array including Terrain Ruggedness Index (TRI).
    
    Parameters:
    array_2d (numpy.ndarray): Input 2D array (interpreted as elevation data for TRI)
    
    Returns:
    dict: Dictionary containing all calculated statistics
    """
    # Flatten the array for some calculations
    flattened = array_2d.flatten()
    
    # Initialize results dictionary
    results = {}
    
    # 1. Mean
    results['mean'] = np.mean(array_2d)
    
    # 2. Variance
    results['variance'] = np.var(array_2d)
    
    # 3. Kurtosis (Fisher's definition, normal = 0.0)
    results['kurtosis'] = stats.kurtosis(flattened, fisher=True)
    
    # 4. Skewness
    results['skewness'] = stats.skew(flattened)
    
    # 5. Terrain Ruggedness Index (TRI)
    # Calculate the difference between each cell and its 8 neighbors
    rows, cols = array_2d.shape
    tri_values = np.zeros_like(array_2d)
    
    for i in range(rows):
        for j in range(cols):
            # Get all 8 neighbors (using zero padding for edges)
            neighbors = []
            for x in [-1, 0, 1]:
                for y in [-1, 0, 1]:
                    if x == 0 and y == 0:
                        continue  # skip the center cell
                    if 0 <= i+x < rows and 0 <= j+y < cols:
                        neighbors.append(array_2d[i+x, j+y])
            
            if neighbors:  # only calculate if there are neighbors
                # Calculate squared differences from center cell
                squared_diffs = [(array_2d[i,j] - n)**2 for n in neighbors]
                tri_values[i,j] = np.sqrt(np.mean(squared_diffs))
    
    results['TRI'] = np.mean(tri_values)
    
    # 6. Wasserstein metric (Earth Mover's Distance)
    # For 2D array, we'll compare row-wise distributions to their mean
    mean_distribution = np.mean(array_2d, axis=0)
    wasserstein_distances = []
    for row in array_2d:
        # Normalize the distributions to make them comparable
        row_norm = row / np.sum(row) if np.sum(row) != 0 else row
        mean_norm = mean_distribution / np.sum(mean_distribution) if np.sum(mean_distribution) != 0 else mean_distribution
        w_dist = wasserstein_distance(row_norm, mean_norm)
        wasserstein_distances.append(w_dist)
    results['wasserstein_metric'] = np.mean(wasserstein_distances)
    
    return results

stats_results = calculate_statistics(ValueTerrain)
for stat, value in stats_results.items():
    print(f"{stat:20}: {value:.4f}")
        
