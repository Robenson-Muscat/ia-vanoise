library(terra)

folder <- "aoi_chm/"
files_tif <- list.files(folder, pattern = "\\.tif$", full.names = TRUE)
raster_list <- list()

#
for (file in files_tif) {
  chm_file <- rast(file)
  raster_list <- c(raster_list, list(chm_file))
}

#do.call to merge all rasters in the list
merged_chm <- do.call(mosaic, raster_list)


copy_merged_chm<-merged_chm


merged_chm[copy_merged_chm <= 0] <- 0
merged_chm[copy_merged_chm > 0 & copy_merged_chm <= 0.5] <- 1
merged_chm[copy_merged_chm > 0.5 & copy_merged_chm <= 1.5] <- 2
merged_chm[copy_merged_chm > 1.5 & copy_merged_chm <= 3] <- 3
merged_chm[copy_merged_chm > 3] <- 4

writeRaster(merged_chm, file.path(folder, "AOI_chm.tif")) 



