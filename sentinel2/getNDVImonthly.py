import openeo
import json
import numpy as np
import matplotlib.pyplot as plt
import rasterio
import rioxarray
import glob
import os
import scipy

# Function to sort filenames alphanumerically
def alphanumeric_sort(name):
    parts = re.split('(\d+)', name)  
    return [int(part) if part.isdigit() else part for part in parts] 


def plot_NDVI(file_path, ax, title):
    with rasterio.open(file_path) as src:
        image = src.read(1)
        ax.imshow(image, cmap='RdYlGn')
        ax.set_title(title)
        ax.axis('off')




connection = openeo.connect(url="openeo.dataspace.copernicus.eu")
connection.authenticate_oidc()

## Coordinates of our AOI(Area of Interest)

fields_aoi = json.loads(
    """
{
  "type": "FeatureCollection",
  "features": [
    {
      "type": "Feature",
      "properties": {},
      "geometry": {
        "type": "Polygon",
        "coordinates": [
          [
            [6.6544450, 45.2258959],
            [6.6544450, 45.2641447],
            [6.6751340, 45.2641447],
            [6.6751340, 45.2258959]
          ]
        ]
      }
    }
  ]
}"""
)


s2cube = connection.load_collection(
    "SENTINEL2_L2A",
    temporal_extent=["2023-06-27", "2023-09-30"],
    spatial_extent=fields_aoi,
    bands=["B04", "B03", "B02", "B08","SCL"],
    max_cloud_cover=20
)







intervals=[["2023-05-01", "2023-05-31"], ["2023-06-27", "2023-06-30"],["2023-07-01", "2023-07-07"],["2023-08-01", "2023-08-31"],["2023-09-02", "2023-09-30"]]




#Cloud masking
scl = s2cube.band("SCL")
mask = ((scl == 8) | (scl == 9))

g = scipy.signal.windows.gaussian(11, std=1.6)
kernel = np.outer(g, g)
kernel = kernel / kernel.sum()

mask = mask.apply_kernel(kernel)
mask = mask > 0.1
s2cube_masked = s2cube.mask(mask)





#Filter dates
red = s2cube_masked.band("B04")
nir = s2cube_masked.band("B08")
ndvi_masked = (nir - red) / (nir + red)
ndvi_monthly_masked = ndvi_masked.aggregate_temporal(intervals=intervals,reducer="mean")



# Download  data
job = ndvi_monthly_masked.execute_batch(
    outputfile=None,  # 
    format="GTIFF"
)
job.get_results().get_assets()
job.get_results().download_files(target="./final_ndvi_monthly_masked/")



download_dir ="../final_ndvi_monthly_masked"
files_tif= glob.glob(os.path.join(download_dir, "*.tif"))
sorted_filenames = sorted(files_tif, key=alphanumeric_sort)



# Displaying the monthly NDVI TIFF files
fig, axes = plt.subplots(nrows=1, ncols=5, figsize=(20, 5))
month_names = ["May", "June", "July", "August", "September"]

for i, ax in enumerate(axes.flatten()):
    file_path = sorted_filenames[i]
    
    plot_NDVI(file_path, ax, month_names[i])

plt.tight_layout()
plt.show()




#Reproject in the following CRS :2154
target_crs = "EPSG:2154"
for file in files_tif:

    img = rioxarray.open_rasterio(file)
    current_crs = img.rio.crs
    reprojected_img = img.rio.reproject(target_crs)
    
    output_file = file.replace(".tif", "_reprojected_2154.tif")
    reprojected_img.rio.to_raster(output_file)
    print(f"Reprojected file saved as: {output_file}")
