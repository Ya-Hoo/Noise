import numpy as np
import json


def initValueGrid(N: int) -> np.ndarray:
    """Initialize a 2D grid of random values using vectorized operations

    Args:
        N (int): side length of map

    Returns:
        np.ndarray: 2D grid of random values
    """
    np.random.seed(30042603)
    return np.random.random((N + 1, N + 1))  # +1 for grid points
    
def smoothstep(t: np.ndarray, n: int) -> np.ndarray:
    """Vectorized smoothstep function

    Args:
        t (np.ndarray): array representation of the map
        n (int): order of the smoothstep function

    Returns:
        np.ndarray: interpolated array
    """
    if n == 3:  # order = 3
        return -2 * t**3 + 3 * t**2
    elif n == 5:  # order = 5
        return 6 * t**5 - 15 * t**4 + 10 * t**3
    return t  # Linear interpolation

def Value(
        xcoords: np.ndarray, 
        ycoords: np.ndarray, 
        values: np.ndarray, 
        λ: float, 
        A: float, 
        smooth: int
    ) -> np.ndarray:
    """Compute a single octave of noise using vectorized operations

    Args:
        xcoords (np.ndarray): list of x-coordinates
        ycoords (np.ndarray): list of y-coordinates
        values (np.ndarray): grid of random values
        λ (float): frequency
        A (float): amplitude
        smooth (int): smoothstep function to be used

    Returns:
        np.ndarray: single octave of noise
    """
    # Scale coordinates by frequency
    nx = xcoords * λ
    ny = ycoords * λ
    
    # Get integer and fractional parts
    x0 = np.floor(nx).astype(int)
    y0 = np.floor(ny).astype(int)
    x_frac = nx - x0
    y_frac = ny - y0
    
    # Apply smoothstep to fractional parts
    u = smoothstep(x_frac, smooth)
    v = smoothstep(y_frac, smooth)
    
    # Get neighboring grid points (with wrapping)
    x1 = x0 + 1
    y1 = y0 + 1
    grid_size = values.shape[0]
    
    # Bilinear interpolation
    n00 = values[y0 % grid_size, x0 % grid_size]
    n10 = values[y0 % grid_size, x1 % grid_size]
    n01 = values[y1 % grid_size, x0 % grid_size]
    n11 = values[y1 % grid_size, x1 % grid_size]
    
    nx0 = (1 - u) * n00 + u * n10
    nx1 = (1 - u) * n01 + u * n11
    return A * ((1 - v) * nx0 + v * nx1)


def terrainGen(params) -> np.ndarray:
    """Generate terrain by combining multiple noise octaves

    Args:
        params (Dict): noise parameters

    Returns:
        np.ndarray: octave value noise
    """
    N = params["N"]
    values = initValueGrid(N)
    
    # Precompute normalized coordinates [0,1)
    y_coords, x_coords = np.mgrid[0:N, 0:N] / N
    
    terrain = np.zeros((N, N), dtype=np.float32)
    λ = params["FREQUENCY"]
    A = params["AMPLITUDE"]
    
    for _ in range(params["OCTAVES_CNT"]):
        terrain += Value(x_coords, y_coords, values, λ, A, params["SMOOTHSTEP"])
        λ *= params["LACUNARITY"]
        A *= params["PERSISTENCE"]
    
    t_min, t_max = np.min(terrain), np.max(terrain)
    if t_max > t_min:  # Avoid division by zero for flat terrain
        return (terrain - t_min) / (t_max - t_min)
    return np.zeros_like(terrain)


# Load parameters and compute
with open("parameters.json", "r") as f:
    params = json.load(f)
    terrain = terrainGen(params)