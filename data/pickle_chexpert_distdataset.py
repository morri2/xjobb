
from datahandlers import DistDataset
from torch.utils.data import DataLoader
import pickle
import os

dirname = os.path.dirname(__file__)
print(dirname, type(dirname))
print("building new dataset - unpickling failed")
cxr_imgs_dataset = DistDataset( dirname + "/cheXpert/cxr_imgs{:03}.pt")
with open(dirname + "/cheXpert/cxr_imgs_dataset.pickle", "wb") as f:
    pickle.dump(cxr_imgs_dataset, f)
