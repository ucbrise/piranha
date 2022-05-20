#!/usr/bin/env python
# coding: utf-8

# # Dataset generator
# This will code will be used to output datasets (MNIST or CIFAR10) in " " separated formated.

# ### Setup

# In[1]:

import os
import random
import torch
import numpy as np
from torchvision import transforms, datasets
from torch.utils.data import DataLoader

torch.manual_seed(0)
random.seed(0)
np.random.seed(0)

# Let us build some helpers. Function to get dataset as numpy array from a loader, a function to save data to file with space delimiter, and a function to convert labels into a vector of one hot labels.

# In[2]:

def get_dataset(loader):
  images, labels = [], []
  for img, label in loader:
    images.append(img)
    labels.append(label)
  return torch.cat(images).numpy(), torch.cat(labels).numpy()


def save_to_file(tensor, filename):
    np.savetxt(fname=filename, delimiter=" ", X=tensor.flatten().tolist())
    

def one_hot(labels):
    one_hot_labels = np.zeros((labels.size, 10))
    one_hot_labels[np.arange(labels.size),labels] = 1
    return one_hot_labels

# ## CIFAR10 Dataset

TARGET_DATASET = datasets.CIFAR10
BATCH_SIZE = 128
path = "../files/CIFAR10/"

train_loader = DataLoader(TARGET_DATASET("./", train=True, transform=transforms.ToTensor(), download=True), batch_size=BATCH_SIZE, shuffle=False, drop_last=True)
test_loader = DataLoader(TARGET_DATASET("./", train=False, transform=transforms.ToTensor(), download=True), batch_size=BATCH_SIZE, shuffle=False, drop_last=True)

train_images, train_labels = get_dataset(train_loader)
test_images, test_labels = get_dataset(test_loader)

# Save the images to file and convert the labels into one hot before saving. Repeat for test data.

train_images = np.transpose(train_images, [0, 3, 2, 1])
print(train_images.shape)
print(one_hot(train_labels).shape)

save_to_file(train_images, path+"train_data")
save_to_file(one_hot(train_labels), path+"train_labels")

test_images = np.transpose(test_images, [0, 3, 2, 1])
print(test_images.shape)
print(one_hot(test_labels).shape)

save_to_file(test_images, path+"test_data")
save_to_file(one_hot(test_labels), path+"test_labels")

