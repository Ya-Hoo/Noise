import rasterio  # For GeoTIFF reading
import numpy as np
from skimage.transform import resize

# SWISS ALPES
with rasterio.open('geoData/everest.tiff') as src:
    everest = src.read(1)  # Read first band
    bounds = src.bounds      # Geographic bounds
    resolution = src.res    # Pixel size in degrees

# Resample to 256x256
everest_resampled = resize(
    everest, 
    (256, 256), 
    order=1  # 1=bilinear, 3=cubic
)

everest_normalized = (everest_resampled - np.min(everest_resampled)) / (np.max(everest_resampled) - np.min(everest_resampled))

# HIMALAYAN
with rasterio.open('geoData/denali.tiff') as src:
    denali = src.read(1)  # Read first band
    bounds = src.bounds      # Geographic bounds
    resolution = src.res    # Pixel size in degrees

# Resample to 256x256
denali_resampled = resize(
    denali, 
    (256, 256), 
    order=1  # 1=bilinear, 3=cubic
)

denali_normalized = (denali_resampled - np.min(denali_resampled)) / (np.max(denali_resampled) - np.min(denali_resampled))

# ROCKY
with rasterio.open('geoData/matterhorn.tiff') as src:
    matterhorn = src.read(1)  # Read first band
    bounds = src.bounds      # Geographic bounds
    resolution = src.res    # Pixel size in degrees

# Resample to 256x256
matterhorn_resampled = resize(
    matterhorn, 
    (256, 256), 
    order=1  # 1=bilinear, 3=cubic
)

matterhorn_normalized = (matterhorn_resampled - np.min(matterhorn_resampled)) / (np.max(matterhorn_resampled) - np.min(matterhorn_resampled))
                      