# Imitation Learning Neural Network

The network takes in images and labels of manually driven vehicles and learns to predict actions for new images.

## Setup Process

Navigate to /net and create a directory called /data. Insert the [Dataset](https://drive.google.com/file/d/1acWm7MnHFfuF7rvhrm_kfzEZ9dyZ6hmL/view?usp=sharing) after unzipping it.

Run Trainer.ipynb to train the neural network.
Saved model versions will be saved in the snapshots folder

In order to monitor the loss you will have to install tensorboard with the following command:
pip install tensorboard

In order to see the update in the loss you can view the results on Tensorboard by:
1) Opening a new terminal window
2) run 'tensorboard --logdir runs' in the terminal
3) Navigating to the link displayed on the terminal

# Important
While we don't have gitignore set up, don't push commits with the dataset in the file structure, there's no point of having the dataset backed up on github.

# Online Report
[google doc](https://drive.google.com/drive/folders/1LK9MIhBG0G2vXhcsIcVYs7jsulb1ul9L?usp=sharing)
