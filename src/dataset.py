import os
import torch
import numpy as np
from torchvision import transforms

class dataset:
    def __init__(self, root, transforms=transforms, test=False):
        self.test = test
        if self.test:
            npy_files = [os.path.join(root + "/ya_test", f) for f in os.listdir(root + "/ya_test") if f.endswith('.npy')]
            self.csv_files = npy_files
        else:
            npy_files = [os.path.join(root + "/ya_train", f) for f in os.listdir(root + "/ya_train") if f.endswith('.npy')]
            self.csv_files = npy_files
        self.csv_files = npy_files
        self.augment = transforms.Compose([
            transforms.RandomHorizontalFlip(),
            transforms.RandomVerticalFlip(),
            transforms.RandomRotation(30)
])


    def __getitem__(self, index):
        csv_path = self.csv_files[index]
        data = np.load(csv_path)
        data = torch.tensor(data, dtype=torch.float32)
        data = data.permute(2, 0, 1)  # C x H x W
        lonlat = data[:2, :, :]


        if not self.test:
            data[2:, :, :] = self.augment(data[2:, :, :])  
        inp = data[2:31, :, :]
        label = data[31, :, :]
        mask = data[32, :, :]

        return lonlat, inp, label, mask

    def __len__(self):
        return len(self.csv_files)



if __name__ == "__main__":
    dataset_path = "Data"
    dataset1 = dataset(dataset_path, test=False)
    print(len(dataset1))