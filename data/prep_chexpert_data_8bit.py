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
import matplotlib.pyplot as plt


# Dataset is split over 23 files.
# Count dataset size
dirname = os.path.dirname(__file__)

transform = transforms.Compose([   
        #transforms.ToTensor(),
        transforms.CenterCrop(320),
    ])

def transformed_image(n, data): # returns transformed nth image of data
    img_bytes = data.loc[n]['image']['bytes']

    

    with open(dirname + "/img-temp/temp.jpeg", "wb") as f:
        f.write(img_bytes)
    
    pil_img = Image.open(dirname + "/img-temp/temp.jpeg")
    #print("pil size", pil_img.size)
    np_img = np.array(pil_img.getdata(), dtype=np.uint8)
    
    np_img = np_img.reshape((pil_img.size[1], pil_img.size[0]))
    #print("np img shape", np_img.shape)
    

    if np_img.shape[0] < 320 or np_img.shape[1] < 320:
        return None
    #print(np_img.dtype)
    #print(np_img.max())
    pt_img = torch.from_numpy(np_img,)

    #print(pt_img.dtype, pt_img.shape)
    #print("np img shape", np_img.shape)

    pt_img_crop = transform(pt_img)

    # fig, ax = plt.subplots(4)
    # ax[0].imshow(pil_img, cmap="gray")
    # ax[1].imshow(np_img, cmap="gray")
    # ax[2].imshow(pt_img, cmap="gray")
    # ax[3].imshow(pt_img_crop, cmap="gray")
    # plt.show()


    return pt_img_crop


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
        if pt_img is None:
            continue
        images.append(pt_img)
        cxr_count += 1
    image_tensor = torch.stack(images)
    print("  image stack shape", image_tensor.shape)
    out_path = dirname + "/cheXpert_8bit/cxr_imgs{:03}.pt".format(j)
    torch.save(image_tensor, out_path)
    print("  saved to ", out_path)
    images = []

print("\nDONE!")
print("  #cxrs= ", cxr_count)
