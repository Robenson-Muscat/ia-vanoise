
library(lidR)
library(sf)


folder <- "aoi_dal/"
folder_aoi <- "aoi_dal_part/"


files_laz <- list.files(folder, pattern = "\\.laz$", full.names = TRUE)

#Function to split a LAZ file(square of 1km²) into four squares of 0.25km²
split_las_in_four <- function(file_path, output_folder) {

  las <- readLAS(file_path)
  
  if (is.empty(las)) {
    stop("The LAS file is empty or could not be read")
  }
  
  # Get the spatial limits of the LAS file
  las_extent <- st_bbox(las)

  xmin<-las_extent[1]
  ymin<-las_extent[2]
  xmax<-las_extent[3]
  ymax<-las_extent[4]
  
  
  xmid <- (xmin + xmax) / 2
  ymid <- (ymin + ymax) / 2
  
  
  las1 <-  clip_rectangle(las, xmin, ymin, xmid, ymid)
  las2 <-  clip_rectangle(las, xmid, ymin, xmax, ymid)
  las3 <-  clip_rectangle(las, xmin, ymid, xmid, ymax)
  las4 <-  clip_rectangle(las, xmid, ymid, xmax, ymax)
  
  
  writeLAS(las1, file.path(output_folder, paste0(basename(file_path), "_part1.laz")))
  writeLAS(las2, file.path(output_folder, paste0(basename(file_path), "_part2.laz")))
  writeLAS(las3, file.path(output_folder, paste0(basename(file_path), "_part3.laz")))
  writeLAS(las4, file.path(output_folder, paste0(basename(file_path), "_part4.laz")))
}



for (file in files_laz) {
  split_las_in_four(file, folder_aoi)
}


