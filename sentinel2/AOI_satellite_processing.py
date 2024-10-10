
import openeo
import json
import matplotlib.pyplot as plt
import rioxarray
import scipy
import xarray



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
    temporal_extent=["2023-05-01", "2023-09-30"],
    spatial_extent=fields_aoi,
    bands=["B04", "B03", "B02", "B08","SCL"],
    max_cloud_cover=20
)






#Visualization
s2cube.download("load-raw-20cloud.nc")
ds = xarray.load_dataset("load-raw-20cloud.nc")
data = ds[["B04", "B03", "B02"]].to_array(dim="bands")
num_dates = data.sizes['t']

fig, axes = plt.subplots(ncols=num_dates, figsize=(48, 16), dpi=90, sharey=True)
if num_dates == 1:
    axes = [axes]

for i in range(num_dates):
    data[{"t": i}].plot.imshow(vmin=0, vmax=2000, ax=axes[i])
    
plt.savefig("AOI_cover20.png") 
plt.show()



#Cloud masking
scl = s2cube.band("SCL")
mask = ((scl == 8) | (scl == 9))

g = scipy.signal.windows.gaussian(11, std=1.6)
kernel = np.outer(g, g)
kernel = kernel / kernel.sum()

mask = mask.apply_kernel(kernel)
mask = mask > 0.1
s2cube_masked = s2cube.mask(mask)





#Visualization after Cloud Masking
s2cube_masked.download("load-raw-20cloud-masked.nc")
ds = xarray.load_dataset("load-raw-20cloud-masked.nc")
data = ds[["B04", "B03", "B02"]].to_array(dim="bands")
num_dates = data.sizes['t']

fig, axes = plt.subplots(ncols=num_dates, figsize=(48, 16), dpi=90, sharey=True)
if num_dates == 1:
    axes = [axes]

for i in range(num_dates):
    data[{"t": i}].plot.imshow(vmin=0, vmax=2000, ax=axes[i])
  

plt.savefig("AOI_cover20_mask.png") 
plt.show()





#Images from 01/06 to 27/06, 01/09, 08/07 are not masked enough