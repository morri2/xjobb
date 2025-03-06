# util: handles datasets distributed over multiple files
import torch
from torch.utils.data import Dataset, DataLoader
import os


class DistDataset(Dataset):
    def __init__(self, file_path_fmt, max_file_count=100):
        # file_path_fmt is the formatable string of the path to a file in the dataset will be formated with the index of the file
        self.file_path_fmt = file_path_fmt
        
        t0 = torch.load(self.file_path_fmt.format(0))
        
        self.data_per_file = t0.shape[0]

        self.data_count_in_file = []

        for i in range(max_file_count):
            try:
                d = torch.load(self.file_path_fmt.format(i))
                self.data_count_in_file.append(d.shape[0])
                print("{} datapoints from  {}".format(d.shape[0], self.file_path_fmt.format(i)))
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


        print("idx={}: loafing {}th datapoint from {}".format(idx_from_zero, idx, self.file_path_fmt.format(fi)))
        d = torch.load(self.file_path_fmt.format(fi))
        return d[idx]
        


if __name__ == "__main__":
    dd = DistDataset("../cheXpert/cxp_cxrs{:03}.pt")
    print(dd[9000])
    print(len(dd))