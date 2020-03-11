
from CustomDataset import ControlsDataset

import os
import torch
from skimage import io, transform
import numpy as np
import matplotlib.pyplot as plt
from torch.utils.data import Dataset, DataLoader
import torch.nn as nn
import torch.nn.functional as F
from tensorboardX import SummaryWriter

# Ignore warnings
import warnings
warnings.filterwarnings("ignore")

plt.ion()   # interactive mode

# Regression Model
class ConvNet(nn.Module):
    def __init__(self, color_channels, outputs, dataset):
        super(ConvNet, self).__init__()

        self.channels = color_channels
        self.report_period = 20
        self.writer = SummaryWriter
        self.start_epoch = 0
        self.infotype = dataset.labels.infotype
        self.dataset = dataset
        self.dataloader = dataset.dataloader

        self.optimizer = None

        img_size = torch.Size([1, color_channels, 170, 640])

        self.convLayers = nn.Sequential(
            nn.Conv2d(color_channels, 24, 5, stride=2),  # in_channels, out_channels, kernel_size,
            nn.ReLU(),
            nn.Conv2d(24, 36, 5, stride=2),
            nn.ReLU(),
            nn.Conv2d(36, 48, 5, stride=2),
            nn.ReLU(),
            nn.Conv2d(48, 64, 3),
            nn.ReLU(),
            nn.Conv2d(64, 64, 3),
            nn.Dropout(0.25) # Dropping elements with a 25% chance
        )

        self.FC = nn.Sequential(
            nn.Linear(in_features=64*2*33, out_features=100),
            nn.ReLU(),
            nn.Linear(in_features=100, out_features=50),
            nn.ReLU(),
            nn.Linear(in_features=50, out_features=10),  # output features was 10
            nn.Linear(in_features=10, out_features=1)
        )

    def forward(self, input):
        input = input.view(input.size(0), 3, 70, 320)
        input = self.convLayers(input)
        print(input.shape)

        input = input.view(input.size(0), -1)

        output = self.FC(input)
        return output

    def accuracy(self, predictions, target):
        totalCorrect = 0
        for i in range(len(predictions)):
            if predictions[i] == target[i]:
                totalCorrect += 1

        return totalCorrect/len(target)

    def fit(self, device, epochs, optimizer, criterion):
        self.train()
        self.optimizer = optimizer

        trainLoss = 0

        for epoch in range(self.start_epoch, epochs):

            for iBatch, sampledBatch in enumerate(self.dataloader):

                images = sampledBatch['image'].to(device).float()
                controls = sampledBatch['control'].to(device).float()
                controls = torch.flatten(controls)

                optimizer.zero_grad()

                predictions = self(images)
                predictions = torch.flatten(predictions)
                # print(predictions)
                # print(controls)

                loss = criterion(predictions, controls)
                print("Loss", loss)
                loss.backward()
                optimizer.step()



                print("Accuracy", self.accuracy(predictions, controls))
                # trainLoss += loss.data[0].item()

                # if iBatch % self.report_period == 0:
                #     print("Loss:", loss)


# Classification Model
class ClassConvNet(nn.Module):
    def __init__(self, color_channels, outputs, dataset):
        super(ClassConvNet, self).__init__()

        self.channels = color_channels
        self.report_period = 20
        self.writer = SummaryWriter
        self.start_epoch = 0
        self.infotype = dataset.labels.infotype
        self.dataset = dataset
        self.dataloader = dataset.dataloader

        self.optimizer = None

        self.convLayers = nn.Sequential(
            nn.Conv2d(color_channels, 24, 5, stride=2),  # in_channels, out_channels, kernel_size,
            nn.ReLU(),
            nn.Conv2d(24, 36, 5, stride=2),
            nn.ReLU(),
            nn.Conv2d(36, 48, 5, stride=2),
            nn.ReLU(),
            nn.Conv2d(48, 64, 3),
            nn.ReLU(),
            nn.Conv2d(64, 64, 3),
            nn.Dropout(0.25)
        )

        self.FC = nn.Sequential(
            nn.Linear(in_features=64*2*33, out_features=100),
            nn.ReLU(),
            nn.Linear(in_features=100, out_features=50),
            nn.ReLU(),
            nn.Linear(in_features=50, out_features=25),  # output features was 10
            nn.Linear(in_features=25, out_features=outputs),
            nn.ReLU()
        )

    def forward(self, input):
        input = input.view(input.size(0), self.channels, 70, 320)
        input = self.convLayers(input)
        input = input.view(input.size(0), -1)
        output = self.FC(input)
        return output

    def accuracy(self, predictions, target):
        totalCorrect = 0
        for i in range(len(predictions)):
            values, indices = torch.max(predictions[i], 0)
            if indices == target[i]:
                totalCorrect += 1

        return (totalCorrect/len(target)) * 100

    def fit(self, device, epochs, optimizer, criterion):
        self.train()
        self.optimizer = optimizer

        trainLoss = 0


        for epoch in range(self.start_epoch, epochs):

            for iBatch, sampledBatch in enumerate(self.dataloader):

                images = sampledBatch['image'].to(device).float()
                controls = sampledBatch['control'].to(device).long()
                controls = torch.flatten(controls)
                optimizer.zero_grad()


                # print("Controls", controls)
                # exit(1)

                predictions = self(images)
                # print("Predictions", predictions)
                loss = criterion(predictions, controls)
                loss.backward()
                optimizer.step()

                print("[%d, %d] Loss:" %(epoch, iBatch), loss)
                print("Accuracy", self.accuracy(predictions, controls))


class EnsemblesNet(nn.Module):

    def __init__(self, modelList, color_channels, classes, dataset):
        super(EnsemblesNet, self).__init__()

        self.models = modelList
        self.classes = classes

        self.channels = color_channels
        self.report_period = 20
        self.writer = SummaryWriter
        self.start_epoch = 0
        self.infotype = dataset.labels.infotype
        self.dataset = dataset
        self.dataloader = dataset.dataloader

        self.optimizer = None

    def forward(self, input):

        outputs = []
        for i in range(len(self.models)):
            out = self.models[i].forward(input)
            outputs.append(out)

        










