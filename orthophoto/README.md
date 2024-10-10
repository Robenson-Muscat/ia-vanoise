

We have used a pretrained Self Supervised Learning model, based on DINOV2, which performs to work on aerial images and particulary "for High resolution Canopy Height Prediction inference" (https://github.com/facebookresearch/HighResCanopyHeight). We will use it to get embeddings(vectors that represents an image in a high-dimensional space. Each number corresponds to a feature : an attribute of the image) of aerial images(224px224p, which is the suitable size for our model) of an Area of Interest. Then, we are going to cluster them using Random Forest following a training into 3 classes (Forest, Mineral class and Lawn).
  
 


## Data source

Orthophotos of Orgere's valley


## Requirements 


### Install QGIS
To assess the clustering

### Python packages 

pytorch, 

pytorch lightning, 

pandas 

  

Example of successful environment creation for inference 

  

``` 

conda create -n hrch python=3.9 -y 

conda activate hrch 

conda install pytorch==2.0.1 torchvision==0.15.2 pytorch-cuda=11.7 -c pytorch -c nvidia 

pip install pytorch_lightning==1.7  

pip install pandas 

pip install matplotlib 

pip install torchmetrics==0.11.4 

pip install functools

pip install geopandas

``` 

### Install aws cli 

``` 

curl "https://awscli.amazonaws.com/awscli-exe-linux-x86_64.zip" -o "awscliv2.zip" 

unzip awscliv2.zip 

sudo ./aws/install 

``` 

 

 ## Clone the repository 

``` 

git clone https://github.com/facebookresearch/HighResCanopyHeight 

 ``` 

## Data and pretrained models 

  

You can download the data and saved checkpoints from  

``` 

s3://dataforgood-fb-data/forests/v1/models/ 

``` 

### Data 

  
To prepare the data, in the cloned repository, run these commands: 

``` 

aws s3 --no-sign-request cp --recursive s3://dataforgood-fb-data/forests/v1/models/ . 

unzip data.zip 

rm data.zip 

``` 

## Build-up our model



``` python

import sys 
sys.path.insert(0,'HighResCanopyHeight') #ou sys.path.append 
 
 


import torch 
from PIL import Image 
from torchvision import transforms 
import torchvision.transforms as T 
import glob 
from tqdm import tqdm 
import torch.utils.data as data 
import torch.nn as nn 
import pytorch_lightning as pl 
 
from models.backbone import SSLVisionTransformer 

 

class SSLembed(nn.Module):  
    def __init__(self): 
        super().__init__() 
        self.backbone = SSLVisionTransformer(out_indices=16, embed_dim=1280, 
                                             num_heads=20, 
                                             depth=32, 
                                             pretrained=None) 
        print(self.backbone) 
         
    def forward(self, x): 
        x = self.backbone(x) 
        return x 
   
class SSLModule(pl.LightningModule): 
    def __init__(self,  
                  ssl_path="HighResCanopyHeight/saved_checkpoints/compressed_SSLhuge.pth"): 
        super().__init__() 
        self.chm_module_ = SSLembed().eval()  
        ckpt = torch.load(ssl_path, map_location='cpu') 
        self.chm_module_ = torch.quantization.quantize_dynamic( 
            self.chm_module_,  
            {torch.nn.Linear,torch.nn.Conv2d,  torch.nn.ConvTranspose2d}, 
            dtype=torch.qint8) 
        self.chm_module_.load_state_dict(ckpt, strict=False) 
        self.chm_module = lambda x: 10*self.chm_module_(x) 
         
    def forward(self, x): https://arxiv.org/abs/2305.10472

```

### Extract tif subimages from a tiff image (extract_subimg.py) 

this script reads a raster image, slices it into 224x224 pixel sub-images, and saves each sub-image in a specified folder with a file name indicating its coordinates in the original image. 

### Extract an area of interest from a Tiff image, specifying the coordinates (extract_aoi.py) 
This code extracts a specific part of a geospatial image using geographic coordinates and saves it as a new raster file. 

In particular, the function uses the `windows.from_bounds` function, which creates a reading window using the coordinates supplied and the geographic transformation of the source file. 

### Shapefile creation with geometries (generate_shp.py)
In short, this script reads tif images, generates polygons representing patches of these images(224px224p, which is the suitable size for our model), sorts the polygons according to their geographic coordinates, and saves the sorted polygons in a shapefile. 

The code is parallelized using the ` ProcessPoolExecutor ` function in the `functools` package. 

 





## Documentation


https://github.com/facebookresearch/HighResCanopyHeight

[[`Very high resolution canopy height maps from RGB imagery using self-supervised vision transformer and convolutional decoder trained on Aerial Lidar`] (https://arxiv.org/abs/2304.07213)]



[[`Nine tips for ecologists using machine learning`] (https://arxiv.org/abs/2305.10472)]