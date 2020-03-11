import os
import sys
import torch
from skimage import io, transform
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
import torch.nn as nn
import pandas as pd
import torch.optim as optim
import math
import scipy.stats as stats

from CustomDataset import ControlsDataset
from Model import ConvNet, ClassConvNet

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
device = 'cpu'
print('using device', device)
torch.cuda.empty_cache()

# dataset = ControlsDataset()
# # dataset.labels.num_categories = 21
# # dataset.labels.transform.categorize(1.05, -1.05, 21)
# dataset.images.set_grayscale(False)
# # dataset.images.transform.resize(2, dataset.images)
#
# print("Data size", len(dataset.labels))
# dataset.make_dataloaders()


def trainClassification(categories=14, canny=False):

    dataset = ControlsDataset()
    dataset.images.transform.add("resize")
    if canny:
        dataset.images.transform.add("lineHighlight")

    dataset.labels.num_categories = categories
    dataset.labels.transform.categorize(1.05, -1.05, categories)
    # pd.set_option('display.max_rows', None)

    print(dataset.labels.dataframe)
    exit(1)
    dataset.make_dataloaders()

    net = ClassConvNet(color_channels=3, outputs=categories, dataset=dataset).to(device)
    # net = ConvNet(color_channels = 1, outputs = 21, dataset = dataset).to(device)
    print("number of parameters: ", sum(p.numel() for p in net.parameters()))

    epoch = 30
    optimizer = optim.Adam(net.parameters(), lr=0.001)
    criterion = nn.CrossEntropyLoss()
    net.report_period = 20
    net.optimizer = optimizer

    net.fit(device, epoch, optimizer, criterion)


def trainRegression(canny=False):
    dataset = ControlsDataset()
    dataset.images.transform.add("resize")
    if canny:
        dataset.images.transform.add("lineHighlight")

    dataset.make_dataloaders()

    net = ConvNet(color_channels=3, outputs=1, dataset=dataset).to(device)
    # net = ConvNet(color_channels = 1, outputs = 21, dataset = dataset).to(device)
    print("number of parameters: ", sum(p.numel() for p in net.parameters()))

    epoch = 30
    optimizer = optim.Adam(net.parameters(), lr=0.001)
    criterion = nn.MSELoss()
    net.report_period = 20
    net.optimizer = optimizer

    net.fit(device, epoch, optimizer, criterion)


# trainRegression()
trainClassification()
