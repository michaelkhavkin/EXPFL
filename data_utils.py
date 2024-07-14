from torch.utils.data import Dataset, DataLoader
import numpy as np

class ImageDataset(Dataset):
    def __init__(self, images_series, labels, transform=None):
        self.images_series = images_series
        self.labels = labels
        self.transform = transform

    def __len__(self):
        return len(self.images_series)

    def __getitem__(self, idx):
        image = self.images_series.iloc[idx]
        if self.transform:
            image = self.transform(image)
        label = self.labels[idx]

        return image, label
    
def preprocess_img(img):
    imp = img.numpy().transpose((1, 2, 0))
    mean = np.array([0.485, 0.456, 0.406])
    std = np.array([0.229, 0.224, 0.225])
    imp = std * imp + mean
    imp = np.clip(imp, 0, 1)
    return imp