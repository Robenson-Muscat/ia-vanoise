
import os
import geopandas as gpd
from shapely.geometry import Polygon
import numpy as np
import concurrent.futures
import functools
import warnings
import re
import pandas as pd
import rioxarray

# Function to sort filenames alphanumerically
def alphanumeric_sort(name):
    parts = re.split('(\d+)', name)  
    return [int(part) if part.isdigit() else part for part in parts] 

# Function to generate shapefile for a single image
def generate_shapefile(path, filename):
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        file_path = os.path.join(path, filename)
        img = rioxarray.open_rasterio(file_path)
    bounds = img.rio.bounds()
    num_patches_x = img.rio.width // 16
    num_patches_y = img.rio.height // 16

    polys = []
    for j in range(num_patches_y):
        for i in range(num_patches_x):
            left = bounds[0] + i * 16 * img.rio.resolution()[0]
            top = bounds[3] - j * 16 * np.abs(img.rio.resolution()[1])
            right = left + 16 * img.rio.resolution()[0]
            bottom = top - 16 * np.abs(img.rio.resolution()[1])
            poly = Polygon([(left, top), (right, top), (right, bottom), (left, bottom), (left, top)])
            polys.append((poly, filename)) 
    return polys

# Function to generate shapefiles for multiple images in parallel
def generate_shapefile_parallel(path, output):
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        filenames = [filename for filename in os.listdir(path) if filename.endswith('.tif')]
        polys_with_filenames = []
        sorted_filenames = sorted(filenames, key=alphanumeric_sort)
        with concurrent.futures.ProcessPoolExecutor(max_workers=14) as executor:
            future_to_filename = {executor.submit(generate_shapefile, path, filename): filename for filename in sorted_filenames}
            for future in concurrent.futures.as_completed(future_to_filename):
                filename = future_to_filename[future]
                try:
                    polys_with_filenames.extend(future.result())
                except Exception as exc:
                    print(f'Error processing {filename}: {exc}')
        
        polys, filenames = zip(*polys_with_filenames)
        gdf = gpd.GeoDataFrame({'geometry': polys, 'source_file': filenames})
        gdf.to_file(output)

#Sort by y-coordinate (north to south) then by x-coordinate (west to east)
def sort_by_northwest(gdf):
    return gdf.sort_values(by=['centroid_y', 'centroid_x'], ascending=[False, True])







imagepath = "aoi_subimg/"
output_shapefile = "clustering_aoi_newsupervised.shp"
generate_shapefile_parallel(imagepath, output_shapefile)



#Sort the shapefile in the following order: 
#geographical position of the sub-image from which the patch is extracted, latitude(north to south), longitude(west to east) 
gp=gpd.read_file(output_shapefile)
gp_sorted_fil = gp.sort_values(by='source_fil', key=lambda x: x.map(alphanumeric_sort))

gp_sorted_fil['centroid'] = gp_sorted_fil.geometry.centroid
gp_sorted_fil['centroid_x'] = gp_sorted_fil.centroid.x
gp_sorted_fil['centroid_y'] = gp_sorted_fil.centroid.y




# Apply the sort
sorted_gdf_list = []
for i in range(0, len(gp_sorted_fil['source_fil'].unique())):
    chunk = gp_sorted_fil.iloc[i*196:(i+1)*196]
    sorted_chunk = sort_by_northwest(chunk)
    sorted_gdf_list.append(sorted_chunk)


gp_sorted = gpd.GeoDataFrame(pd.concat(sorted_gdf_list, ignore_index=True))
gp_sorted=gp_sorted.drop(columns=['centroid_x','centroid_y','centroid'])






