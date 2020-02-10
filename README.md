# Imitation Learning Neural Network

The network takes in images and labels of manually driven vehicles and learns to predict actions for new images.

## Setup Process

Navigate to /net and insert the [Dataset](https://drive.google.com/file/d/1acWm7MnHFfuF7rvhrm_kfzEZ9dyZ6hmL/view?usp=sharing) after unzipping it.

Run Trainer.ipynb to train the neural network.
Saved model versions will be saved in the snapshots folder

In order to see the update in the loss you can view the results on Tensorboard by:
1) Opening a new terminal window
2) run 'tensorboard - -logdir runs'
3) Navigating to the link displayed on the terminal
