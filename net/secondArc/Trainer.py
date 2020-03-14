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
from Model import ConvNet, ClassConvNet, EnsemblesNet, ClassConvNetNorm

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
device = 'cpu'
print('using device', device)
torch.cuda.empty_cache()


def check():
    dataset = ControlsDataset()
    dataset.images.transform.add("resize")
    dataset.labels.num_categories = 14
    dataset.labels.transform.categorize(1.05, -1.05, 14)

    dataset.make_dataloaders()

    net = ClassConvNet(3, 14, dataset)
    activation = {}

    def get_activation(name):
        def hook(model, input, output):
            activation[name] = output.detach()

        return hook

    net.convLayers.register_forward_hook(get_activation('convLayers'))

    image = dataset[0]['image']

    image = torch.Tensor(image).to(device).float()

    image = image.reshape(1, 3, 70, 320)
    output = net(image)

    print("output", output.shape)

    act = activation['convLayers'].squeeze()
    print("activationShape", act.shape)
    plt.figure()
    fig, axarr = plt.subplots(nrows=4, ncols=4)

    for x in range(4):
        for y in range(4):
            axarr[y][x].imshow(act[4 * x + y], cmap='gray')


def layerAnalysis():
    dataset = ControlsDataset()
    dataset.images.transform.add("resize")
    dataset.labels.num_categories = 14
    dataset.labels.transform.categorize(1.05, -1.05, 14)

    dataset.make_dataloaders()

    net = ClassConvNet(color_channels=3, outputs=14, dataset=dataset).to(device)

    activation = {}

    def get_activation(name):
        def hook(model, input, output):
            activation[name] = output.detach()

        return hook

    net.convLayers.register_forward_hook(get_activation('convLayers'))

    image = dataset[0]['image']
    image = torch.Tensor(image).to(device).double()
    image = torch.tensor(image.reshape(1, 3, 70, 320))
    output = net.convLayers(image)
    print(output.shape)
    print(type(image))
    print(image.shape)

    return
    image = torch.Tensor(image).to(device).long()
    image = image.reshape(1, 3, 70, 320)
    output = net(image)

    act = activation['convLayers'].squeeze()

    fig, axarr = plt.subplots(act.size(0))
    io.imshow(act[0])
    io.show()
    # for idx in range(act.size(0)):
    #     axarr[idx].imshow(act[idx])
    #
    # plt.show()
    # fig, axarr = plt.subplots(nrows=4, ncols=4)
    # for x in range(4):
    #     for y in range(4):
    #         axarr[y][x].imshow(act[4 * x + y], cmap='gray')


def trainClassification(categories=14, canny=False, grayScale=False):
    dataset = ControlsDataset()
    dataset.images.transform.add("resize")
    inputChannels = 3
    if canny:
        dataset.images.transform.add("lineHighlight")

    if grayScale:
        dataset.images.transform.add("grayscale")
        dataset.images.set_grayscale(True)


    if dataset.images.grayscale or canny:
        inputChannels = 1


    dataset.labels.num_categories = categories
    dataset.labels.transform.categorize(1.05, -1.05, categories)

    dataset.make_dataloaders()

    net = ClassConvNet(color_channels=inputChannels, outputs=categories, dataset=dataset).to(device)

    print("number of parameters: ", sum(p.numel() for p in net.parameters()))

    # print("Labels", dataset.labels.dataframe['Category'])
    # exit(1)

    epoch = 50
    optimizer = optim.Adam(net.parameters(), lr=0.0001)
    criterion = nn.CrossEntropyLoss()
    net.report_period = 20
    net.optimizer = optimizer

    net.fit(device, epoch, optimizer, criterion)

def trainClassification2(categories=14, canny=False, grayScale=False):
    dataset = ControlsDataset()
    dataset.images.transform.add("resize")
    inputChannels = 3
    if canny:
        dataset.images.transform.add("lineHighlight")

    if grayScale:
        dataset.images.transform.add("grayscale")
        dataset.images.set_grayscale(True)


    if dataset.images.grayscale or canny:
        inputChannels = 1

    dataset.labels.num_categories = categories
    dataset.labels.transform.categorize(1.05, -1.05, categories)


    dataset.make_dataloaders()

    net = ClassConvNetNorm(color_channels=inputChannels, outputs=categories, dataset=dataset).to(device)

    print("number of parameters: ", sum(p.numel() for p in net.parameters()))

    epoch = 50
    optimizer = optim.Adam(net.parameters(), lr=0.0001)
    criterion = nn.CrossEntropyLoss()
    net.report_period = 20
    net.optimizer = optimizer

    net.fit(device, epoch, optimizer, criterion)


# TODO Need to configure the setup, Not enough time
# TODO Need to figure out a way to a variety of dataset
def trainEnsemblesClassifier(categories=14, canny=False):
    dataset = ControlsDataset()
    dataset.images.transform.add("resize")
    inputChannels = 3

    dataset.labels.num_categories = categories
    dataset.labels.transform.categorize(1.05, -1.05, categories)
    dataset.make_dataloaders()

    eModel = EnsemblesNet(classes=categories, color_channels=inputChannels, dataset=dataset).to(device)

    # Building Models 1 Normal
    net1 = ClassConvNet(color_channels=inputChannels, outputs=categories, dataset=dataset).to(device)
    eModel.addModel(net1)

    # Building Model 2
    net2 = ClassConvNet(color_channels=inputChannels, outputs=categories, dataset=dataset).to(device)
    eModel.addModel(net2)

    # Training
    epoch = 30
    optimizer = optim.Adam(eModel.parameters(), lr=0.0001)
    criterion = nn.CrossEntropyLoss()
    eModel.report_period = 20
    eModel.optimizer = optimizer

    eModel.fit(device, epoch, optimizer, criterion)


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


# TODO Train each of these models on Epoch 50
# Models To Train
trainClassification()
# trainClassification(canny=True)
# trainClassification(grayScale=True)
#
# trainClassification2()
# trainClassification2(canny=True)
# trainClassification2(grayScale=True)

