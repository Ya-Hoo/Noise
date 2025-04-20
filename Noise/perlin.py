import numpy as np
import json


def initGradientGrid(N: int) -> np.ndarray:
    """Initialise a 2D grid of random unit vectors using vectorized cos/sin

    Args:
        N (int): side length of map

    Returns:
        np.ndarray: 2D grid of random unit vectors
    """
    angles = np.random.uniform(0, 2 * np.pi, size=(N + 1, N + 1))
    gradients = np.stack([np.cos(angles), np.sin(angles)], axis=-1)
    return gradients.astype(np.float32)
    
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

def Perlin(
        xcoords: np.ndarray, 
        ycoords: np.ndarray, 
        gradients: np.ndarray, 
        λ: float, 
        A: float, 
        smooth: int
    ) -> np.ndarray:
    """Compute a single octave of noise using vectorized operations

    Args:
        xcoords (np.ndarray): list of x-coordinates
        ycoords (np.ndarray): list of y-coordinates
        values (np.ndarray): grid of random unit vectors
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
    
    # Get neighboring grid points (with wrapping)
    x1 = x0 + 1
    y1 = y0 + 1
    
    # Get gradients at grid points (with wrapping)
    grid_size = gradients.shape[0]
    grid_size = gradients.shape[0]
    g00 = gradients[y0 % grid_size, x0 % grid_size]
    g10 = gradients[y0 % grid_size, x1 % grid_size]
    g01 = gradients[y1 % grid_size, x0 % grid_size]
    g11 = gradients[y1 % grid_size, x1 % grid_size]
    
    # Compute dot products
    d00 = (x_frac) * g00[..., 0] + (y_frac) * g00[..., 1]
    d10 = (x_frac - 1) * g10[..., 0] + (y_frac) * g10[..., 1]
    d01 = (x_frac) * g01[..., 0] + (y_frac - 1) * g01[..., 1]
    d11 = (x_frac - 1) * g11[..., 0] + (y_frac - 1) * g11[..., 1]
    
    # Apply smoothstep
    u = smoothstep(x_frac, smooth)
    v = smoothstep(y_frac, smooth)
    
    # Bilinear interpolation
    nx0 = (1 - u) * d00 + u * d10
    nx1 = (1 - u) * d01 + u * d11
    return A * ((1 - v) * nx0 + v * nx1)

def terrainGen(params) -> np.ndarray:
    """Generate terrain by combining multiple noise octaves

    Args:
        params (Dict): noise parameters

    Returns:
        np.ndarray: octave Perlin noise
    """
    N = params["N"]
    gradients = initGradientGrid(N)
    
    # Precompute normalized coordinates [0,1)
    y_coords, x_coords = np.mgrid[0:N, 0:N] / N
    
    terrain = np.zeros((N, N), dtype=np.float32)
    λ = params["FREQUENCY"]
    A = params["AMPLITUDE"]
    
    for _ in range(params["OCTAVES_CNT"]):
        terrain += Perlin(x_coords, y_coords, gradients, λ, A, params["SMOOTHSTEP"])
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