
import json
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import openeo



## Coordinates of our AOI(Area of Interest)
fields = json.loads(
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
            [6.581, 45.297],
            [6.729, 45.297],
            [6.729, 45.218],
            [6.581, 45.218]
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
    bands=["B04", "B08"],
)

red = s2cube.band("B04")
nir = s2cube.band("B08")
ndvi = (nir - red) / (nir + red)

timeseries = ndvi.aggregate_spatial(geometries=fields, reducer="mean")

job = timeseries.execute_batch(out_format="CSV", title="NDVI timeseries")

job.get_results().download_file("ndvi-results/timeseries-basic.csv")
pd.read_csv("ndvi-results/timeseries-basic.csv", index_col=0).head()




def plot_timeseries(filename, figsize=(6, 3)):
    # 
    df = pd.read_csv(filename, index_col=0)
    # 
    df.index = pd.to_datetime(df.index)
    # 
    df.sort_index(inplace=True)

    print(df.index)

    #
    fig, ax = plt.subplots(figsize=figsize, dpi=90)

    # 
    for name, group in df.groupby("feature_index"):
        group["avg(band_0)"].plot(marker="o", ax=ax, label=name)

    ax.set_ylabel("NDVI")
    ax.set_ylim(-1, 1)

    

    
    plt.show()

# 
plot_timeseries("ndvi-results/timeseries-basic.csv")
