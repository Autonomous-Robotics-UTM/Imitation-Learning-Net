import os
import torch
import pandas as pd
from skimage import io, transform, viewer
import numpy as np
import matplotlib.pyplot as plt
from torch.utils.data import Dataset, DataLoader, SubsetRandomSampler
from imgAugmentors import ImageTransform, LabelTransform
import math

# Ignore warnings
import warnings

warnings.filterwarnings("ignore")
plt.ion()  # interactive mode


# Class to extract out images and preform different augmentations for training
class Images:
    def __init__(self, path):
        self.path = path
        self.transform = ImageTransform()
        self.stack_size = 1  # TODO remove stack size, doesn't seem to be used
        self.grayscale = False

    # Returns the corresponding image name
    def image_filename(self, path, number):
        return "{0}{number:06}.jpg".format(path, number=number)

    # Applies Normalization and returns a stack_size of images as single input
    def get_stack(self, index, stack_size):
        img_filenames = [self.image_filename(self.path, i)
                         for i in range(index, index + stack_size)]
        images = np.array([io.imread(img_filename).transpose((2, 0, 1)) for img_filename in img_filenames])
        images = self.transform.apply(images) / 255
        if not self.grayscale:
            images = np.concatenate(images, axis=0)
        return images

    def get_image_size(self):
        initialImage = self.image_filename(self.path, 0)
        image = io.imread(initialImage)
        print(image.shape)

    def showSingleImage(self):
        initialImage = self.image_filename(self.path, 0)
        image = io.imread(initialImage)
        io.imshow(image)
        io.show()

    # Used to display stack images
    def show(self, stack):
        stack = stack.squeeze()
        if self.grayscale:
            f, ax = plt.subplots(self.stack_size, 1, figsize=(1 * self.stack_size, 25))
            ax.imshow(stack, cmap='gray')

        else:
            f, ax = plt.subplots(self.stack_size, 3, figsize=(3 * self.stack_size, 25))
            ax = ax.reshape((self.stack_size, 3))
            for k in range(3 * self.stack_size):
                i, j = k // 3, k % 3
                ax[i, j].imshow(stack[k], cmap='gray')


    # Enables grayScale image transformation
    def set_grayscale(self, switch):
        if self.grayscale != switch:
            self.grayscale = switch
            if switch:
                self.transform.add("grayscale")
            else:
                self.transform.remove("grayscale")


class Labels:
    def __init__(self, path):
        # Reading in data csv from the specific path
        self.dataframe = pd.read_csv(path)
        self.transform = LabelTransform(self)
        self.override_params = True

        # categorical attributes (Max steering angles
        # 1.05 ==> Right, -1.05 =+> Left
        self.maximum = 1.05
        self.minimum = -1.05

        # directional attributes
        self.directions = {
            "left": [[-1.05, 0]],
            "straight": [[-0.01, 0.01]],
            "right": [[0, 1.05]]}

        self.infotype = "Angle"

    # Length of the data (ie rows
    def __len__(self):
        return len(self.dataframe)

    # Getting a data point at the specifeid input
    def __getitem__(self, index):
        return self.dataframe.iloc[index][self.infotype]

    # Data bar representation
    def histogram(self):
        return self.dataframe[self.infotype].hist()


# Builds the main dataset, Has access to both images and labels
# Main Class used when data is required
class ControlsDataset(Dataset):
    """Dataset that maps camera images into steering angle"""
    def __init__(self, stack_size=1, img_folder='../data/cropped_reduced/', csv_path='../data/cropped_reduced/data.csv'):
        self.stack_size = stack_size
        self.images = Images(img_folder)
        self.labels = Labels(csv_path)

    # Builds the training and testing datasets based on the given percentages
    def make_dataloaders(self, train=0.8, test=0.2):
        indices = list(range(len(self)))
        split = int(np.floor(test * len(self)))

        # spliting the dataset
        train_indices, val_indices = indices[split:], indices[:split]

        # Creating PT data samplers and loaders:
        train_sampler = SubsetRandomSampler(train_indices)
        valid_sampler = SubsetRandomSampler(val_indices)

        # Training data loader # NOTE had to remove shuffle
        self.dataloader = DataLoader(self, batch_size=16, num_workers=0, sampler=train_sampler)

        # Validation data loader # NOTE had to remove shuffle
        self.validloader = DataLoader(self, batch_size=16, num_workers=0, sampler=valid_sampler)

        print("Total training stacks", len(self.dataloader))
        print("Total validation stacks", len(self.validloader))

    # Reads in images and the corresponding control inputs
    # Combines the images and the controls in a single dataframe
    def __getitem__(self, idx):
        # get a stack of images
        image_stack = self.images.get_stack(idx, self.stack_size)

        # use the latest image as the control
        # label = self.labels[idx + self.stack_size]
        # label = np.array([label])
        label = self.labels.dataframe['Category'][idx + self.stack_size]
        label = np.array([label])


        # combine stack and label together
        sample = {'image': image_stack,
                  'control': label}
        return sample

    def __len__(self):
        return len(self.labels) - self.stack_size




if __name__ == "__main__":
    print("Preforming tests")
    stack_size = 1
    dataset = ControlsDataset(stack_size)
    dataset.labels.num_categories = 21
    dataset.labels.transform.categorize(1.05, -1.05, 21)

    print('Column Labels', list(dataset.labels.dataframe.columns))
    print('Total Labels', dataset.labels.__len__())
    print("Getting Single Image")

    print("Categoried Data", dataset.labels.dataframe)


    dataset.images.showSingleImage()

    # io.set_title("Original Image. (NO TRANSFORMATION")

    dataset.images.transform.add("resize")
    dataset.images.transform.add("lineHighlight")

    images = dataset[0]['image']
    print("Image Shape", images.shape)
    io.imshow(images)
    io.show()

    # io.show()
    # io.set_title("Resizing Applied")

    '''
    
    # Applying Transformation
    dataset.images.transform.add("grayscale")
    dataset.images.set_grayscale(True)
    dataset.images.transform.apply(dataset.images)



    images = dataset[0]['image']
    io.imshow(images[0,:,:])
    io.show()
    io.set_title("Grayscale Applied")

    # Resizing

    images = transform.resize(images, (images.shape[0], 85, 320))
    io.imshow(images[0, :, :])
    io.show()
    
    '''

    # dataloader = DataLoader(dataset, batch_size=4, shuffle=True, num_workers=0)



    counts = dataset.labels.dataframe.groupby('Category')['ID'].count()
    print(counts)
    print(max(counts))
    dataset.labels.histogram()
    print("length, width, channels")

