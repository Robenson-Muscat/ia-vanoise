library(lidR)


folder <- "aoi_dal_part/"
folder_aoi <- "aoi_chm/"

files_laz <- list.files(folder, pattern = "\\.laz$", full.names = TRUE)

for (file in files_laz) { 

    las <- readLAS(file)
    dtm_tin <- rasterize_terrain(las, res = 1, algorithm = tin())

    #normalizes the LAS points to the ground by subtracting the DTM
    nlas <- las - dtm_tin 
    
    rm(las)
    rm(dtm_tin)
    chm <- rasterize_canopy(nlas, res = 1, algorithm = p2r())
    

    
    name_file <- tools::file_path_sans_ext(basename(file))
    path_file_raster <- paste0(folder_aoi, name_file, "_chm.tif")
  
    terra::writeRaster(chm, path_file_raster, 
                     filetype = "GTiff", overwrite = TRUE, NAflag = NA)

    rm(chm)
    
    gc()

}



