import os
import torch
import pandas as pd
from skimage import io, transform, feature
import numpy as np
import matplotlib.pyplot as plt
from torch.utils.data import Dataset, DataLoader, SubsetRandomSampler
import math

# Ignore warnings
import warnings
warnings.filterwarnings("ignore")
plt.ion()   # interactive mode


class ImageTransform:
    def __init__(self):
        self.transformations = []

    # Applying all the transformation
    def apply(self, images):
        for transformation in self.transformations:
            function = getattr(self, transformation)
            images = function(images)
        return images

    # Function to add in a new transformation
    def add(self, name):
        self.transformations.append(name)

    # Removing a previous transformation
    def remove(self, name):
        if name in self.transformations:
            self.transformations.remove(name)

    # Default grayscale transformation
    def grayscale(self, images):
        if images.shape[1] != 3:
            return images
        r, g, b = images[:, 0, :, :], images[:, 1, :, :], images[:, 2, :, :]
        gray = 0.2989 * r + 0.5870 * g + 0.1140 * b
        return gray

    def resize(self, images):
        height = 70
        width = 320
        images = transform.resize(images, (images.shape[0], images.shape[1], height, width))
        return images

    def lineHighlight(self, image):
        gImage = self.grayscale(image)
        # print(gImage.shape)
        withOutChannel = gImage[0, :, :]
        lineImage = feature.canny(withOutChannel)
        gImage[0, :, :] = lineImage
        return gImage


class LabelTransform:
    # Takes in the initial labels
    def __init__(self, labels):
        self.labels = labels.dataframe

    # Method to modify the values of a specific column in the
    # Data file
    def column_rule(self, column_name, function):
        self.labels[column_name] = self.labels.apply(lambda x: function(x.Angle), axis=1)

    # Converts angle into their specific categories
    def categorize(self, maximum, minimum, num_categories):

        # Transformation function for the angles column
        def func(angle):
            if angle == maximum:
                return num_categories - 1
            scale = num_categories / (maximum - minimum)
            return math.floor(scale * (angle - minimum))

        # apply the categorization to the column
        self.column_rule("Category", func)

        # return the categories created
        num_range = np.linspace(minimum, maximum, num_categories + 1)
        categories = [[round(num_range[i], 3), round(num_range[i + 1], 3)] for i in range(len(num_range) - 1)]

        return categories

    # directionalize, gives the corresponding direction
    # key based on the angle
    def directionalize(self, directions):
        def direction(angle):
            for key in directions:
                for interval in directions[key]:
                    start, end = interval
                    if start < angle and end > angle:
                        return key

        self.labels['Direction'] = self.labels.apply(lambda x: direction(x.Angle), axis=1)


