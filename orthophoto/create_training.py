import os
import numpy as np
import rasterio
from rasterio.windows import Window

# Folder of selected images for training
output_folder = 'new_training'
if not os.path.exists(output_folder):
    os.makedirs(output_folder)

# Read coordinates of images for training
coordinates_file = 'new_coordinates.txt'
coordinates = []
with open(coordinates_file, 'r') as file:
    for line in file:
        if line.strip():  # 
            try:
                lon, lat = map(float, line.strip().split(','))
                coordinates.append((lon, lat))
            except ValueError:
                print(f"Ignored : {line.strip()}")






filename = '../Orgere/RGB_orgere.tif'
with rasterio.open(filename) as src:
    for i, (lon, lat) in enumerate(coordinates):
        
        row, col = src.index(lon, lat)
        
        half_size = 112  
        window = Window(col - half_size, row - half_size, 224, 224)
        
        sub_image = src.read(window=window)
        
        output_path = os.path.join(output_folder, f'new_sub_image_{i+1}.tif')
        with rasterio.open(
            output_path,
            'w',
            driver='GTiff',
            height=224,
            width=224,
            count=src.count,
            dtype=sub_image.dtype,
            crs=src.crs,
            transform=rasterio.windows.transform(window, src.transform),
        ) as dst:
            dst.write(sub_image)
