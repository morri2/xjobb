
import os
import torch
from torch.utils.data import Dataset, DataLoader
from torchvision.transforms import transforms
import math
import pickle

import matplotlib.pyplot as plt

class SplitDataset(Dataset):
    def __init__(self, dataset, split_start=0, split_end=None):
        self.dataset = dataset
        self.split_start = (int)(split_start)
        self.split_end = (int)(split_end) if split_end is not None else len(dataset)

        if self.split_end > len(self.dataset):
            print("!!! SplitDataset - Split End too large")


    def __len__(self):
        return self.split_end - self.split_start
    
    def __getitem__(self, index):
        index += self.split_start
        if index >= self.split_end or index < self.split_start:
            return None
        return self.dataset[index]


class DistDataset(Dataset):
    def __init__(self, file_path_fmt, max_file_count=100, cache_last_file=True):
        # file_path_fmt is the formatable string of the path to a file in the dataset will be formated with the index of the file
        self.file_path_fmt = file_path_fmt
        print("# building dataset")
        
        t0 = torch.load(self.file_path_fmt.format(0))
        
        self.data_per_file = t0.shape[0]

        self.data_count_in_file = []

        self.last_file_idx = None
        self.last_file_data = None

        self.cache_last_file = cache_last_file

        for i in range(max_file_count):
            try:
                d = torch.load(self.file_path_fmt.format(i))
                self.data_count_in_file.append(d.shape[0])
                print(" - {} datapoints from  {}".format(d.shape[0], self.file_path_fmt.format(i)))
            except:
                break
            
    def __len__(self):
        return sum(self.data_count_in_file)

    
    def __getitem__(self, idx):
        idx_from_zero = idx
        fi = 0
        while idx >= self.data_count_in_file[fi]:
            idx -= self.data_count_in_file[fi]
            fi += 1


        #print("idx={}: loafing {}th datapoint from {}".format(idx_from_zero, idx, self.file_path_fmt.format(fi)))
        
        if self.cache_last_file: 
            if self.last_file_idx is not None and self.last_file_data is not None:
                if self.last_file_idx == fi:
                    return self.last_file_data[idx]
        
        
        
        d = torch.load(self.file_path_fmt.format(fi))

        if self.cache_last_file:
            self.last_file_data = d
            self.last_file_idx = fi

        return d[idx]
   

def gauss_noise_signal_dependent(img: torch.Tensor, sd=0.3):
    # sd is standard deviation where the pixel value is 1.0
    img = img.clamp(0.01, 1.0)
    noisy_img = torch.normal(img, img * sd)
    return torch.clip( noisy_img, min=0.0, max=1.0)

def gauss_noise(img: torch.Tensor, sd=0.3):
    img = img.clamp(0.01, 1.0)
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

        
        if self.img_extract_fn is None:
            image: torch.Tensor = self.dataset[idx]
        else:
            image : torch.Tensor = self.img_extract_fn(self.dataset[idx])


        if image.dtype == torch.uint8: # make float
            image = image.float()
            image = image / 255.0
            

        # Image noising function application
        image_real = image
        image_noisy = self.noise_fn(image)

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
