from torch.utils.data import Dataset, DataLoader

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