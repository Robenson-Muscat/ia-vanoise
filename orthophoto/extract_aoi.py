import rasterio
from rasterio.windows import from_bounds

def extract_subimage(tiff_path, left, bottom, right, top, output_path):
    
    with rasterio.open(tiff_path) as src:
        # 
        window = from_bounds(left, bottom, right, top, src.transform)
        
        sub_image = src.read(window=window)
        
        out_meta = src.meta.copy()
        out_meta.update({
            "driver": "GTiff",
            "height": window.height,
            "width": window.width,
            "transform": src.window_transform(window)
        })
        
        
        with rasterio.open(output_path, "w", **out_meta) as dest:
            dest.write(sub_image)


filename="../Orgere/RGB_orgere.tif"

#Extract a tiff of our Area of Interest
left=986704.5
bottom=6465143.0
right=988131.9
top=6469395.4

extract_subimage(filename, left, bottom, right, top, "AOI_Orgere.tiff")

#Extract the first quarter 

left = bounds.left + (bounds.right - bounds.left) / 2
right = bounds.right
top = bounds.bottom + (bounds.top - bounds.bottom) / 2
bottom = bounds.bottom

extract_subimage(filename, left, bottom, right, top, "4th_quarter_Orgere.tiff")


#Extract the last 16th
left = bounds.left + 3 * (bounds.right - bounds.left) / 4
right = bounds.right
top = bounds.bottom + (bounds.top - bounds.bottom) / 4
bottom = bounds.bottom

extract_subimage(filename, left, bottom, right, top, "16th_Orgere.tiff")


