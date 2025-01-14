import torch
import torch.nn as nn
import torch.optim as optim
import torchvision
import numpy as np
import time
import math
import copy

from torchvision import transforms
from matplotlib import pyplot as plt
from tqdm import tqdm
from typing import Sequence, Union, Tuple


############################################# ----- LOAD THE DATA ----- ################################################
batch_size = 16
data_root = 'C:\Learning\Computer Science\Machine Learning\\02 Databases Used\CV Datasets\Multi Class Classification Datasets\Fashion MNIST\data\\fashion_mnist'

train_size = 50000
val_size = 10000


# We do the basic transformations on the data
transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.5), (0.5))
])


# We download the datasets, if it does not exist on the local disk (at the specified path)
# First, we download the training set
dataset_train_val = torchvision.datasets.FashionMNIST(
    root=data_root,
    train=True,
    download=True,
    transform=transform
)

# We take the validation set as a part of the training set using the sized set above
train_set, val_set, _ = torch.utils.data.random_split(dataset_train_val,
                                                      [train_size, val_size, len(dataset_train_val) - train_size - val_size])

# Second, we download the test set
test_set = torchvision.datasets.FashionMNIST(
    root=data_root,
    train=False,
    download=True,
    transform=transform
)


# We set the data loader corresponding to each set
train_loader = torch.utils.data.DataLoader(
    train_set,
    batch_size=batch_size,
    shuffle=True,
    num_workers=2
)

val_loader = torch.utils.data.DataLoader(
    val_set,
    batch_size=batch_size,
    shuffle=True,
    num_workers=2
)

test_loader = torch.utils.data.DataLoader(
    test_set,
    batch_size=batch_size,
    shuffle=True,
    num_workers=2
)


classes = ('tshirt', 'trouser', 'pullover', 'dress', 'coat', 'sandal', 'shirt', 'sneaker', 'bag', 'boot')


####################################### ----- THE ARCHITECTURE ----- ###################################################
class ConvAE(nn.Module):
    def __init__(self, channels: int = 1, n_filters: int = 10, kernel_size: int = 3, central_dim = 100, inp_side_len: int = 28, central_side_len: int = 14):
        super().__init__()

        pad = (kernel_size - 1) // 2

        self.encoder = nn.Sequential(
            nn.Conv2d(in_channels=channels, out_channels=n_filters, kernel_size=kernel_size, stride=1, padding=pad),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2),
            nn.Flatten(),
            nn.Linear(in_features=central_side_len**2*n_filters, out_features=central_dim),
            nn.ReLU()
        )

        self.decoder = nn.Sequential(
            nn.Linear(in_features=central_dim, out_features=central_side_len**2*n_filters),
            nn.ReLU(),
            nn.Unflatten(dim=1, unflattened_size=(n_filters, central_side_len, central_side_len)),
            nn.ConvTranspose2d(in_channels=n_filters, out_channels=channels, kernel_size=2, stride=2, padding=0),
            nn.Sigmoid()
        )


    def forward(self, x):
        encoded = self.encoder(x)
        decoded = self.decoder(encoded)

        return decoded


############################################# ----- TRAINING ----- #####################################################
def compute_epoch_loss(model, data_loader):
    # It computes the loss of the <model> on the dataset's <data_loader> for the current epoch

    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    model = model.to(device)     # Set the model to GPU is available
    model.eval()     # Set the model to evaluation mode

    epoch_loss = 0
    for inputs, labels in tqdm(data_loader):
        inputs, labels = inputs.to(device), labels.to(device)     # Set the data to GPU

        outputs = model(inputs)
        outputs = outputs.to(device)

        loss = criterion(torch.flatten(outputs, 1), torch.flatten(inputs, 1))
        epoch_loss += loss.item()

    epoch_loss = epoch_loss/len(data_loader)

    return epoch_loss


def train(model, train_loader, val_loader, test_loader, num_epochs, criterion, optimizer):
    print("\n\n\n ... The training process ...\n")

    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    model = model.to(device)     # Set the model to GPU if available
    model.train()     # Set the model to the training mode

    best_loss = np.inf
    best_weights = None
    best_epoch = 0

    for epoch in tqdm(range(num_epochs)):
        start_time = time.perf_counter()

        train_loss = 0
        for inputs, labels in tqdm(train_loader):
            inputs, labels = inputs.to(device), labels.to(device)     # Set the data to GPU

            # Forward and Backward passes
            optimizer.zero_grad()
            outputs = model(inputs)
            outputs = outputs.to(device)
            loss = criterion(torch.flatten(outputs, 1), torch.flatten(inputs, 1))
            loss.backward()
            optimizer.step()

            train_loss += loss.item()

        train_loss /= len(train_loader)

        validation_loss = compute_epoch_loss(model=model, data_loader=val_loader)
        if validation_loss < best_loss:
            best_loss = validation_loss
            best_weights = copy.deepcopy(model.state_dict())
            best_epoch = epoch + 1

        end_time = time.perf_counter()
        duration = end_time - start_time

        print(f"Epoch = {epoch + 1} ===> Train Loss = {train_loss: .6f} ===> Time = {duration: .2f} ===> Validation Loss = {validation_loss: .6f} ===> Best Loss = {best_loss: .6f} at epoch {best_epoch}")

    # Set the model('s weights) with the best accuracy
    model.load_state_dict(best_weights)

    test_loss = compute_epoch_loss(model=model, data_loader=test_loader)
    print(f"Test Loss of the Best Model is: {test_loss: .4f}")

    # Save the best model, based on the loss computed on validation set
    path_best_model = "..\\ae_conv.pth"
    torch.save(model, path_best_model)


################################################## ----- MAIN() ----- ##################################################
if __name__ == '__main__':
    start_time = time.perf_counter()

    convolutionalAE = ConvAE(channels=1, n_filters=10, kernel_size=3, central_dim=16, inp_side_len=28, central_side_len=14)

    number_of_epochs = 10
    criterion = nn.MSELoss()
    optimizer = optim.Adam(convolutionalAE.parameters(), lr=1e-3)

    train(model=convolutionalAE, train_loader=train_loader, val_loader=val_loader, test_loader=test_loader,
          num_epochs=number_of_epochs, criterion=criterion, optimizer=optimizer)

    end_time = time.perf_counter()
    duration = end_time - start_time
    print(f"\n\n\nTotal Training Time is: {duration}")
