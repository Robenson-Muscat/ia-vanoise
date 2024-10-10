
Our goal is to create a Canopy Height Model:

![Products](lidarModels.png)

It represents the difference between a digital terrain model and a digital surface model (DSM - DTM = CHM) and gives you the height of objects  on the earth's surface.  This is not an altitude value, but the height or distance from the ground to the top of objects (including trees, or buildings, or any other object detected and recorded by the lidar system). Therefore, we can add diversity into our class 'Lawn'.

## Data source

Download `laz` files here:

https://geoservices.ign.fr/lidarhd


You need to 

- download the DownThemAll extension: https://www.downthemall.net/ which loads extension files (copc.laz extension) 

- Then go to https://diffusion-lidarhd.ign.fr/   

In 2021, the IGN (Institut national de l'information géographique et forestière), as part of the national LiDAR HD program, will begin producing 3D mapping of the entire soil and subsoil of France using LIDAR data. You need to search and select the areas you are looking for, either by clicking on them to select 1km² zones one by one, or by using the Rectangle option to select a set of adjacent 1km² areas. 

- ![Step1](lidr-dal1.png) 

- Then right-click on DownThemAll!-> DownThemAll! :  

![Step2](lidr-dal2.png) 

- A page like this is displayed and all you have to do is click on Download :  

![Step3](lidr-dal3.png) 



## Requirements 

  
`R` package `lidR`

  

``` R

install.packages("lidR")

``` 

## Exemple


### Split laz files into quarters of laz files 

LiDAR files can be very large and difficult to manage in memory. Splitting them into smaller LAZ files makes them easier to manage and create a canopy height model. 

 

### Get a canopy height model (getCHM_aoi.R) 

Process LiDAR files (in .laz format), generate canopy height models (CHM) and save them as raster files (TIFF). 

 

### Merging rasters 

We can merge these CHM rasters on our 0.25 km² squares to obtain the CHM raster over our entire area of interest. 

## Documentation

https://r-lidar.github.io/lidRbook/

https://www.neonscience.org/resources/learning-hub/workshops/work-lidar-derived-rasters-r





