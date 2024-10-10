# https://r-lidar.github.io/lidRbook/chm.html

library(lidR)
#las <- readLAS("LHD_FXX_0985_6467_PTS_O_LAMB93_IGN69.copc.laz")
las <- readLAS("Bauges/LHD_FXX_0951_6508_PTS_C_LAMB93_IGN69.copc.laz")
#plot(las, size = 3, bg = "white")
lasraster <- pixel_metrics(las, mean(Z), res=1)
plot_dtm3d(lasraster, bg = "white")

## Computing a DTM
dtm_tin <- rasterize_terrain(las, res = 1, algorithm = tin())
plot_dtm3d(dtm_tin, bg = "white")

## Showing classification
#plot(las, color = "Classification", size = 3, bg = "white")


## Computing canopy height as the difference between Z and the DTM
nlas <- las - dtm_tin
#plot(nlas, size = 4, bg = "white")
## Rasterizing at resolution 1
chm <- rasterize_canopy(nlas, res = 1, algorithm = p2r())
col <- height.colors(25)
plot(chm, col = col)


# function to plot a crossection of a point cloud
# works with lidR package
library(ggplot2)
plot_crossection <- function(las,
                             p1 = c(min(las@data$X), mean(las@data$Y)),
                             p2 = c(max(las@data$X), mean(las@data$Y)),
                             width = 4, colour_by = NULL)
{
  colour_by <- enquo(colour_by)
  data_clip <- clip_transect(las, p1, p2, width)
  p <- ggplot(data_clip@data, aes(X,Z)) + geom_point(size = 0.5) + coord_equal() + theme_minimal()
  
  if (!is.null(colour_by))
    p <- p + aes(color = !!colour_by) + labs(color = "")
  
  return(p)
}
plot_crossection(las, colour_by = factor(Classification))


## Computing a raster with argmax(Classification) at higher resolution
## Requires averaging
Mode <- function(x) {
  ux <- unique(x)
  return(ux[which.max(tabulate(match(x, ux)))])
}
classificationraster <- pixel_metrics(las, ~Mode(Classification), res=5)
#crs(classificationraster)
plot(classificationraster)
# the lowest available value (given the datatype) is used to represent NA for signed type
# NAflag=NA
terra::writeRaster(classificationraster, "/tmp/classificationraster.tif", filetype = "GTiff", overwrite = TRUE, datatype='INT1U', NAflag=NA)
# this tif image is invisble, because pixels have to be between 0 and 255
# a little trick:
# terra::writeRaster(classificationraster*50, "/tmp/classificationraster.tif", filetype = "GTiff", overwrite = TRUE, datatype='INT1U', NAflag=NA)


# Is canopy height correlated to classification
chm <- rasterize_canopy(nlas, res = 1, algorithm = p2r())
classificationraster <- pixel_metrics(las, ~Mode(Classification), res=1)
boxplot(split(as.matrix(chm), as.matrix(classificationraster)), ylim=c(0,5),
        ylab="estimated canopy height (lidR package)", xlab="IGN classification")
