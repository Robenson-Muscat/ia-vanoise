import numpy as np 
import pandas as pd 
import os
import rioxarray



image_path = "../Orgere/AOI_Orgere.tiff"
img = rioxarray.open_rasterio(image_path)


bounds=img.rio.bounds()
#[0], [1] bounds bottom left corner
#
#[2],[3] bounds upper right corner

def extract_subimages(img, img_origin, save_folder):
    height,width= img.rio.shape
    sub_image_size =  224
    
    
    if not os.path.exists(save_folder):
        os.makedirs(save_folder)
    

    for y in range(0, height - sub_image_size + 1, sub_image_size):
        for x in range(0, width - sub_image_size + 1, sub_image_size):
            
            sub_img = img[:,y:y+sub_image_size, x:x+sub_image_size]

            start_y=y
            start_x=x
            

            

            
            filename = f"subimage_pxl_{start_y}_{start_x}.tif"
            filepath = os.path.join(save_folder, filename)
                
            
            sub_img.rio.to_raster(filepath)


extract_subimages(img, img, "./aoi_subimg/")
