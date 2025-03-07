# tested with python 3.10.13 on nixos
import pandas as pd
import math
from PIL import Image
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import sys
import time
import torch
import numpy as np
from torchvision import datasets, transforms
from torch.utils.data import DataLoader
from torchvision.io import read_image
import os

# Dataset is split over 23 files.
# Count dataset size
dirname = os.path.dirname(__file__)

transform = transforms.Compose([   
        transforms.ToTensor(),
        transforms.CenterCrop(320),
    ])

def transformed_image(n, data): # returns transformed nth image of data
    img_bytes = data.loc[n]['image']['bytes']

    with open(dirname + "/img-temp/temp.jpeg", "wb") as f:
        f.write(img_bytes)
    
    pil_img = Image.open(dirname + "/img-temp/temp.jpeg")
    
    transformed_img = transform(pil_img)

    return transformed_img


images = []

cxr_count = 0

for j in range(0,23):
    fp = dirname + "/cheXpert_hugging_face/train-{:05}-of-00023.parquet".format(j)
    print("...reading", fp)
    data = pd.read_parquet(fp)

    for i in range(data.shape[0]):
        if data.loc[i]['Frontal/Lateral'] == 1:
            #print("skipping lateral cxr")
            continue
        pt_img = transformed_image(i, data)
        images.append(pt_img)
        cxr_count += 1
    image_tensor = torch.stack(images)
    print("  image stack shape", image_tensor.shape)
    out_path = dirname + "/cheXpert/cxr_imgs{:03}.pt".format(j)
    torch.save(image_tensor, out_path)
    print("  saved to ", out_path)
    images = []

print("\nDONE!")
print("  #cxrs= ", cxr_count)
