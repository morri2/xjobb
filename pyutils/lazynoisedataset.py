import torch
from torch.utils.data import Dataset, DataLoader
from torchvision.transforms import transforms
import os
import math
from distrdataset import DistDataset
import pickle

import matplotlib.pyplot as plt

def gauss_noise(img: torch.Tensor, sd=0.3):
    noise = sd * torch.randn(img.shape, dtype=img.dtype, device=img.device, )
    return torch.clip( img + noise, min=0.0, max=1.0)

class LazyNoiseDataset(Dataset):
    def __init__(self, dataset, noise_fn=gauss_noise, size_cap=None, img_extract_fn=None):
        self.dataset = dataset # the dataset of cxr images
        self.noise_fn = noise_fn
        self.img_extract_fn = img_extract_fn # optional function for extracting image from dataset
        self.len = len(self.dataset) if (size_cap == None or size_cap == -1) else min(len(self.dataset), size_cap)

    def from_distdataset_pickle(pickle_path):
        with open(pickle_path, "rb") as f:
            distdataset = pickle.load(f)
        return LazyNoiseDataset(distdataset)


    def __len__(self):
        return self.len

    def __getitem__(self, idx):

        image = None
        if self.img_extract_fn is None:
            image = self.dataset[idx]
        else:
            image = self.img_extract_fn(self.dataset[idx])


        # Image noising function application
        image_real = image.clamp(0.01, 1.0)
        image_noisy = self.noise_fn(image_real)

        return image_noisy, image_real

if __name__ == "__main__":
    
    distdataset = None
    try: 
        print("trying to unpickle dataset")
        with open("pickles/cxrdataset.pickle", "rb") as f:
            distdataset = pickle.load(f)
    except:
        print("building new dataset - unpickling failed")
        distdataset = DistDataset("../cheXpert/cxp_cxrs{:03}.pt")
    
        with open("pickles/cxrdataset.pickle", "wb") as f:
            pickle.dump(distdataset, f)


    lazynoisedataset = LazyNoiseDataset(distdataset)

    with open("pickles/cxrnoisedataset.pickle", "wb") as f:
        pickle.dump(lazynoisedataset, f)


    fig, axs = plt.subplots(4,2)
    for i in range(4):
        a, b = lazynoisedataset[2000 * i]

        axs[i,0].imshow(a.squeeze(), cmap="gray")
        axs[i,1].imshow(b.squeeze(), cmap="gray")
        fig.suptitle("Noisy -vs- CheXpert")


    plt.show()
