import torch
from torch import nn
from torch.utils.data import DataLoader
from torchvision import datasets
from torchvision.transforms import ToTensor
from sklearn.metrics import f1_score
import torch.nn.functional as F
from sklearn.model_selection import KFold
import glob
import os
import nibabel as nib
import numpy as np
from skimage.util import view_as_windows


# Define model
class CNN(nn.Module):
    def __init__(self):
        super().__init__()
        self.convolution = nn.Sequential(
            nn.Conv2d(in_channels=1, out_channels=32, kernel_size=3, padding=1),  # (1, 33, 33) -> (32, 33, 33)
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2), # (32, 33, 33) -> (32, 16, 16)
            nn.Conv2d(32, 32, 3, padding=1),  # (32, 16, 16) -> (32, 16, 16)
            nn.ReLU(),
            nn.MaxPool2d(2)  # (32, 16, 16) -> (32, 8, 8)
        )
        self.flatten = nn.Flatten()
        self.linear_relu_stack = nn.Sequential(
            nn.Linear(32*8*8, 561),
            nn.ReLU(),
            nn.Dropout(0.43269891871173555),
            nn.Linear(561, 23),
            )
    def forward(self, x):
        x = self.convolution(x)
        x = self.flatten(x)
        x = self.linear_relu_stack(x)
        return x


def train(dataloader, model, loss_fn, optimizer,device):
    size = len(dataloader.dataset)
    model.train()
    for batch, (X, y) in enumerate(dataloader):

        X, y = X.to(device), y.to(device)

        # Compute prediction error
        pred = model(X)
        loss = loss_fn(pred, y)

        # Backpropagation
        loss.backward()
        optimizer.step()
        optimizer.zero_grad()

        # if batch % 100 == 0:
        #     loss, current = loss.item(), (batch + 1) * len(X)
        #     print(f"loss: {loss:>7f}  [{current:>5d}/{size:>5d}]")

def test(dataloader, model, loss_fn):
    size = len(dataloader.dataset)
    num_batches = len(dataloader)
    model.eval()
    test_loss, correct = 0, 0
    all_preds = []
    all_labels = []
    with torch.no_grad():
        for X, y in dataloader:
            X, y = X.to(device), y.to(device)
            pred = model(X)
            test_loss += loss_fn(pred, y).item()
            correct += (pred.argmax(1) == y).type(torch.float).sum().item()
            all_preds.extend(pred.argmax(1).cpu().numpy())
            all_labels.extend(y.cpu().numpy())
    test_loss /= num_batches
    correct /= size
    f1_macro = f1_score(all_labels, all_preds, average='macro')
    f1_micro = f1_score(all_labels, all_preds, average='micro')
    f1= f1_score(all_labels, all_preds, average=None)
    print(f"Test Error: \n Accuracy: {(100*correct):>0.1f}%, Avg loss: {test_loss:>8f} \n")
    print(f"F1 macro: {f1_macro:.4f}, F1 micro: {f1_micro:.4f}")
    print(f1)


# K-Fold Training
if __name__ == "__main__":
    all_files = sorted(glob.glob("CTce_ThAb_b33x33_n1000_8bit/*.csv"))
    all_files_basename = [os.path.splitext(os.path.basename(path))[0] for path in all_files]
    # batch_size = 64
    kf = KFold(n_splits=4, shuffle=True, random_state=42)
    data_list = torch.load("data_all.path")
    batch_size = 128
    device = torch.accelerator.current_accelerator().type if torch.accelerator.is_available() else "cpu"
    print(f"Using {device} device")
    model = CNN().to(device)
    model.load_state_dict(torch.load("modelCNNnew.pth"))
    loss_fn = nn.CrossEntropyLoss()
    optimizer = torch.optim.RMSprop(model.parameters(), lr=1.0082099500553373e-6)

    for fold, (train_idx, val_idx) in enumerate(kf.split(all_files_basename)):
        train_files = [all_files_basename[i] for i in train_idx]
        val_files = [all_files_basename[i] for i in val_idx]

        # Lọc data dựa trên source_file
        train_data = [item[:2] for item in data_list if item[2] in train_files]
        val_data = [item[:2] for item in data_list if item[2] in val_files]

        train_loader = DataLoader(train_data, batch_size=64, shuffle=True)
        val_loader = DataLoader(val_data, batch_size=64, shuffle=False)

        print(f"Fold {fold+1}: train={len(train_data)}, val={len(val_data)}")
        epochs = 5
        for t in range(epochs):
            print(f"Epoch {t+1}\n-------------------------------")
            train(train_loader, model, loss_fn, optimizer)
            test(val_loader, model, loss_fn)
    print("Done!")



    torch.save(model.state_dict(), "modelCNNnew.pth")
    print("Saved PyTorch Model State to modelCNNnew.pth")



