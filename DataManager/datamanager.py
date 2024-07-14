import os
import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader
import torchvision.datasets as datasets
import torchvision.transforms as transforms
import requests
from PIL import Image


import warnings

from data_utils import ImageDataset
warnings.filterwarnings("ignore")

def MNIST():
    MNIST_train  = datasets.MNIST(root='../dataset', train=True, download=False, transform=transforms.ToTensor())
    MNIST_test   = datasets.MNIST(root='../dataset', train=False, download=False, transform=transforms.ToTensor())
    return MNIST_train, MNIST_test

def EMNIST():
    EMNIST_train = datasets.EMNIST(root='../dataset', split='balanced', train=True , download=False, transform=transforms.ToTensor())
    EMNIST_test  = datasets.EMNIST(root='../dataset', split='balanced', train=False, download=False, transform=transforms.ToTensor())
    return EMNIST_train, EMNIST_test

def FashionMNIST():
    FashionMNIST_train = datasets.FashionMNIST(root='../dataset', train=True , download=False, transform=transforms.ToTensor())
    FashionMNIST_test  = datasets.FashionMNIST(root='../dataset', train=False, download=False, transform=transforms.ToTensor())
    return FashionMNIST_train, FashionMNIST_test

def ECGHeartbeatCategorization():
    IMG_SIZE = 112
    root_dataset_dir = r'C:/Users/user/Datasets'
    train_df = pd.read_csv("{}/{}/{}".format(dataset_dir_path, r'MIT_BIH', 'mitbih_train.csv.zip'), header=None)
    test_df = pd.read_csv("{}/{}/{}".format(dataset_dir_path, r'MIT_BIH', 'mitbih_test.csv.zip'), header=None)
    X_train = train_df.loc[:,:186]
    y_train = np.array(train_df.loc[:,187:])
    X_test = test_df.loc[:,:186]
    y_test = np.array(test_df.loc[:,187:])

    X_train = X_train.apply(lambda x: Image.fromarray(x.values.reshape(11, 17), 'L'), axis=1)
    X_test = X_test.apply(lambda x: Image.fromarray(x.values.reshape(11, 17), 'L'), axis=1)
    y_train = np.array(y_train).astype(int).squeeze()
    y_test = np.array(y_test).astype(int).squeeze()
    transform = transforms.Compose([transforms.Resize((IMG_SIZE, IMG_SIZE)), transforms.ToTensor(),])
    train_dataset = ImageDataset(X_train, y_train, transform=transform)
    test_dataset = ImageDataset(X_test, y_test, transform=transform) 
    return train_dataset, test_dataset

if __name__ == '__main__':
    MNIST_train = datasets.MNIST(root='../dataset', train=True, download=False, transform=transforms.ToTensor())
    MNIST_test = datasets.MNIST(root='../dataset', train=False, download=False, transform=transforms.ToTensor())

