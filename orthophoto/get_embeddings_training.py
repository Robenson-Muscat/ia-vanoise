import os
import numpy as np
import torch
from PIL import Image
from torchvision import transforms
import torchvision.transforms as T
import glob
from tqdm import tqdm
import torch.utils.data as data
import torch.nn as nn
import pytorch_lightning as pl

import pandas as pd
import re

import sys
sys.path.append("/home/rmondelice/Ortho_vanoise/ssl/HighResCanopyHeight/")
from models.backbone import SSLVisionTransformer




#Sort
def alphanumeric_sort(name):
    parts = re.split('(\d+)', name)  
    return [int(part) if part.isdigit() else part for part in parts] 






class InferenceDataset(data.Dataset):
    def __init__(self, path, transform=None):
        self.filenames = sorted(glob.glob(os.path.join(path, '**/*.png'), recursive=True) + \
                                glob.glob(os.path.join(path, '**/*.jpg'), recursive=True) + \
                                glob.glob(os.path.join(path, '**/*.JPG'), recursive=True) + \
                                glob.glob(os.path.join(path, '**/*.PNG'), recursive=True) + \
                                glob.glob(os.path.join(path, '**/*.jpeg'), recursive=True) + \
                                glob.glob(os.path.join(path, '**/*.JPEG'), recursive=True) + \
                                glob.glob(os.path.join(path, '**/*.tif'), recursive=True))

        self.filenames = sorted(self.filenames, key=alphanumeric_sort)

        self.transform = transform

    def __getitem__(self, index):
        img_path = self.filenames[index]
        img = Image.open(img_path).convert('RGB')
        if self.transform is not None:
            img = self.transform(img)
        return img, img_path

    def __len__(self):
        return len(self.filenames)


class SSLembed(nn.Module): 
    def __init__(self):
        super().__init__()
        self.backbone = SSLVisionTransformer(out_indices=16, embed_dim=1280,
                                             num_heads=20,
                                             depth=32,
                                             pretrained=None)
        
    def forward(self, x):
        x = self.backbone(x)
        return x
  
class SSLModule(pl.LightningModule):
    def __init__(self, 
                  ssl_path="/home/rmondelice/Ortho_vanoise/ssl/HighResCanopyHeight/saved_checkpoints/compressed_SSLhuge.pth"):
        super().__init__()
        self.chm_module_ = SSLembed().eval() 
        ckpt = torch.load(ssl_path, map_location='cpu')
        self.chm_module_ = torch.quantization.quantize_dynamic(
            self.chm_module_, 
            {torch.nn.Linear,torch.nn.Conv2d,  torch.nn.ConvTranspose2d},
            dtype=torch.qint8)
        self.chm_module_.load_state_dict(ckpt, strict=False)
        self.chm_module = lambda x: 10*self.chm_module_(x)
        
    def forward(self, x):
        x = self.chm_module(x)
        return x
    

def predict_and_save(model, dl, device, save_dir, batch_size):
    os.makedirs(save_dir, exist_ok=True)
    filenames, feature_map_avgs = [], []
    count = 0
    for i, batch in tqdm(enumerate(dl), desc='Processing batches'):
        images, files = batch[0], batch[1]
        images = images.to(device)

        with torch.no_grad():
            pred = model.forward(images)
            pred = pred[0][0][0].numpy().reshape([1280,14*14]).transpose()

            feature_map_avgs.append(pred)
            
            filenames += list(files)

        count += 1
        if count % 200 == 0   :
            
        
            feature_map_avgs = np.array(feature_map_avgs)

            feature_map_avgs = feature_map_avgs.reshape(-1, 1280)
            df = pd.DataFrame(data=feature_map_avgs)
            df.to_csv(os.path.join(save_dir, f'embeddings_s{count//200}.csv'), index=False)
            del df
            filenames, feature_map_avgs = [], []

    feature_map_avgs = np.array(feature_map_avgs)

    feature_map_avgs = feature_map_avgs.reshape(-1, 1280)
    df = pd.DataFrame(data=feature_map_avgs)
    df.to_csv(os.path.join(save_dir, f'embeddings_s{count//200+1}.csv'), index=False)    


def getFeatures(path, save_dir):
    model = SSLModule()
    model.eval()
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model.to(device)

    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize((0.420, 0.411, 0.296), (0.213, 0.156, 0.143))
    ])
    batch_size = 1
    dataset = InferenceDataset(path, transform)
    dl = data.DataLoader(dataset, batch_size=batch_size, num_workers=4)

    predict_and_save(model, dl, device, save_dir, batch_size)


def getClasses(directory):
    
    image_paths = []

    
    for root, _, files in os.walk(directory):
        for file in files:
            
            image_paths.append(os.path.join(root, file))

    
    classes = [os.path.basename(os.path.dirname(image)) for image in image_paths]

    # Assign to the 196 patches of the image the same class label
    classes = np.repeat(classes, 196, axis=0)

    return classes


def getImagesPath(path,features_from_images):

    
    images_names = features_from_images.index

    return images_names


image_paths = "./new_training/"
save_dir = "./training_embd_oi16/"
getFeatures(image_paths, save_dir)









