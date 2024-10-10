


The study of Sentinel 2 satellite images enables us to study NDVI with a high degree of precision and thus obtain a plant biomass index and spatio-temporal information on our area of interest.

## Data source



Create an account on https://openeo.dataspace.copernicus.eu/ 

The `openeo` Python package provides a Python interface to the openEO API, enabling us to process remote sensing and Earth observation data via Python. 

Then install the openeo package with `pip install openeo` and import it `import openeo`. 

- ![Step1](step1.png) 

Then use : 

Click on the link to authenticate yourself and this will open a page in your browser  

- ![Step2](step2.png) 

Authenticate 
- ![Step3](step3.png) 

Click Yes and authentication is completed
![Step4](step4.png) 




## Requirements 


  

``` python

pip install xarray 
pip install netcdf4
pip install h5netcdf
pip install openeo

``` 

## Exemple



### Information about the type of satellite

**Command**:
```python
connection.describe_collection("SENTINEL2_L2A")
```
With this command, we get all the metadata on the Sentinel-2 L2A, including the date of the first data available worldwide, the set of spectral bands, etc.

### Selecting a set of satellite images with specific criteria

**Loading data**:
The `load_collection` function allows you to load a DataCube object, which is a set of satellite data. The data added to the data cube can be restricted using the parameters `id`, `spatial_extent`, `temporal_extent`, `bands`, and `max_cloud_cover`. If no data is available for the given extents, a `NoDataAvailable` exception is raised.

- **id**:
  Limits the data to load to the type of satellite we are interested in. In our case, we choose Sentinel-2 L2A. Sentinel-2 data is classified by preprocessing level. Level-2A (atmospherically corrected surface reflectance in cartographic geometry) is processed using the Sen2Cor algorithm to produce Level-2A, which is bottom-of-atmosphere reflectance. Level-2A data is ideal for research activities as it allows for more in-depth analysis without applying additional atmospheric corrections.

- **spatial_extent**:
  Limits the data to load from the collection to the specified polygon(s). We can provide the data in JSON format, or like this:
  ```python
  spatial_extent = {"west": 6.65, "east": 6.67, "north": 45.26, "south": 45.22}
  ```

- **temporal_extent**:
  Limits the data to load from the collection to the specified closed-left time interval. Applies to all temporal dimensions. The interval must be specified as an array with exactly two elements.

- **bands**:
  Adds only the specified bands to the data cube so that bands not matching the list of band names are not available. In our case, for NDVI calculation, the bands `B04` and `B08` are of interest.

- **max_cloud_cover**:
  Limits the data to the imposed cloud cover. This data is an estimate provided by the Copernicus interface. In my opinion, some satellite images, even filtered, seem to have a higher cloud cover. Therefore, it would be interesting to review them.

---




```python

import openeo
import json



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

```

### Visualize satellite images

To have a look at satellite images containing heavy cloud cover. 
```python
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
```

### Cloud masking

We can alleviate this problem by getting rid of those images with cloud cover. 

I've selected satellite images with less than 20% cloud cover from May to September 2023 

![Before mask](AOI_cover20.png) 

  

I've applied a mask function based on the “SCL” band, which is one of the bands provided by Sentinel L2A. This is a pixel-by-pixel classification (based on a spectral band analysis algorithm). More details here: https://sentiwiki.copernicus.eu/web/s2-processing 

![After mask](AOI_cover20_mask.png) 

  

The Cloud classes are classes 8 and 9, represented by `Cloud Medium Probability` and `Cloud High Probability`. The classes are specified here: https://brazil-data-cube.github.io/specifications/bands/SCL.html 

Based on the mask function provided by the copernicus documentation here: https://documentation.dataspace.copernicus.eu/notebook-samples/openeo/NDVI_Timeseries.html#cloud-masking-in-ndvi. 


```python
scl = s2cube.band("SCL")
mask = ((scl == 8) | (scl == 9))

g = scipy.signal.windows.gaussian(11, std=1.6)
kernel = np.outer(g, g)
kernel = kernel / kernel.sum()

mask = mask.apply_kernel(kernel)
mask = mask > 0.1
s2cube_masked = s2cube.mask(mask)
```

### Get NDVI for periods

By applying the formula (B08-B04)/(B04+B08) to our DataCube object, we obtain NDVI data for our time interval, after applying the `mask` function. 

Once the clouds are masked, we use the `aggregate_temporal_period` function, which will aggregate our NDVI data with the `reducer` parameter function (`mean` is a predefined function) over a `period` time interval (`hour`,`day`, `week`, `month` are predefined values) or time intervals. 

We can create our own function to set the `reducer` parameter. 

 

The `execute_batch` function will create a `BatchJob` object, which will then load all the data in the `format` format specified in the parameter.  

We can then download these files in this `format` onto our device by applying `get_results().download_files` to our `BatchJob` object. 

## Documentation


https://processes.openeo.org/






