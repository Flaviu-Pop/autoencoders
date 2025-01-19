import torch
import torch.nn as nn
import torch.optim as optim
import torchvision
import numpy as np
import time
import copy

from torchvision import transforms
from tqdm import tqdm

# ----------------------------------------------------------------------------------------------------------------------
# -------------------------------------------------- LOAD THE DATA -----------------------------------------------------
# ----------------------------------------------------------------------------------------------------------------------
# We set the batch size
batch_size = 16

# We set the root of the Fashion MNIST datasets
data_root = 'C:\Datasets\Fashion MNIST\data\\fashion_mnist'

# We set the size of training and validation sets, restectively
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
                                                      [train_size, val_size,
                                                       len(dataset_train_val) - train_size - val_size])

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


# ----------------------------------------------------------------------------------------------------------------------
# --------------------------------------------- THE ARCHITECTURE -------------------------------------------------------
# ----------------------------------------------------------------------------------------------------------------------
# We apply this autoencoder for images,
# so first we must flatten the image tensor format to one dimensional tensor and,
# at the end of the architecture, we must reshape the resulting tensor to image tensor's format

class AutoEncoder(nn.Module):
    def __init__(self, input_dim: int = 784, hidden_dim: int = 64, bias: bool = True):
        super().__init__()

        assert input_dim > 0, "Pay attention at input dimension"
        assert hidden_dim > 0, "Pay attention at hidden dimension"

        self.encoder = nn.Sequential(
            nn.Flatten(),
            nn.Linear(in_features=input_dim, out_features=hidden_dim, bias=bias),
            nn.ReLU()
        )

        self.decoder = nn.Sequential(
            nn.Linear(in_features=hidden_dim, out_features=input_dim, bias=bias),
            nn.Sigmoid()
        )

    def forward(self, x):
        encoded = self.encoder(x)
        decoded = self.decoder(encoded)

        x = torch.reshape(decoded, [batch_size, 1, 28, 28])

        return x


# ----------------------------------------------------------------------------------------------------------------------
# ------------------------------------------------- TRAINING -----------------------------------------------------------
# ----------------------------------------------------------------------------------------------------------------------
def compute_epoch_loss(model, data_loader, criterion):
    # It computes the loss of the <model> on the dataset's <data_loader> for the current epoch

    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    model = model.to(device)  # Set the model to GPU is available
    model.eval()  # Set the model to evaluation mode

    epoch_loss = 0
    for inputs, labels in tqdm(data_loader):
        inputs, labels = inputs.to(device), labels.to(device)  # Set the data to GPU

        outputs = model(inputs)
        outputs = outputs.to(device)

        loss = criterion(torch.flatten(outputs, 1), torch.flatten(inputs, 1))
        epoch_loss += loss.item()

    epoch_loss = epoch_loss / len(data_loader)

    return epoch_loss


def train(model, train_loader, val_loader, test_loader, num_epochs, criterion, optimizer):
    print("\n\n\n ... The training process ...\n")

    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    model = model.to(device)  # Set the model to GPU if available
    model.train()  # Set the model to the training mode

    best_loss = np.inf
    best_weights = None
    best_epoch = 0

    for epoch in tqdm(range(num_epochs)):
        start_time = time.perf_counter()

        train_loss = 0
        for inputs, labels in tqdm(train_loader):
            inputs, labels = inputs.to(device), labels.to(device)  # Set the data to GPU if available

            # Forward and Backward passes
            optimizer.zero_grad()
            outputs = model(inputs)
            outputs = outputs.to(device)
            loss = criterion(torch.flatten(outputs, 1), torch.flatten(inputs, 1))
            loss.backward()
            optimizer.step()

            train_loss += loss.item()

        train_loss /= len(train_loader)

        validation_loss = compute_epoch_loss(model=model, data_loader=val_loader, criterion=criterion)
        if validation_loss < best_loss:
            best_loss = validation_loss
            best_weights = copy.deepcopy(model.state_dict())
            best_epoch = epoch + 1

        end_time = time.perf_counter()
        duration = end_time - start_time

        print(f"Epoch = {epoch + 1} ===> Train Loss = {train_loss: .6f} ===> Time = {duration: .2f} ===> "
              f"Validation Loss = {validation_loss: .6f} ===> Best Loss = {best_loss: .6f} at epoch {best_epoch}")

    # Set the model('s weights) with the best accuracy
    model.load_state_dict(best_weights)

    test_loss = compute_epoch_loss(model=model, data_loader=test_loader, criterion=criterion)
    print(f"Test Loss of the Best Model (obtained at epoch {best_epoch}) is: {test_loss: .4f}")

    # Save the best model, based on the Loss computed on validation set
    path_best_model = "..\\ae_shallow.pth"
    torch.save(model, path_best_model)


# ----------------------------------------------------------------------------------------------------------------------
# ------------------------------------------------------- MAIN() -------------------------------------------------------
# ----------------------------------------------------------------------------------------------------------------------
if __name__ == '__main__':
    start_time = time.perf_counter()

    input_dimension = 784
    hidden_dimension = 64
    bias = True

    ae = AutoEncoder(input_dim=input_dimension, hidden_dim=hidden_dimension, bias=bias)

    number_of_epochs = 25
    criterion = nn.MSELoss()
    optimizer = optim.Adam(ae.parameters(), lr=1e-3)

    train(model=ae, train_loader=train_loader, val_loader=val_loader, test_loader=test_loader,
          num_epochs=number_of_epochs, criterion=criterion, optimizer=optimizer)

    end_time = time.perf_counter()
    duration = end_time - start_time
    print(f"\n\n\nTotal Training Time: {duration}")
