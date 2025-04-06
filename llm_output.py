import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torchvision import datasets, transforms
from torch.utils.data import DataLoader

class Net(nn.Module):
    def __init__(self, input_channels=1):
        super(Net, self).__init__()
        self.conv1 = nn.Conv2d(input_channels, 32, 3, 1)
        self.conv2 = nn.Conv2d(32, 64, 3, 1)
        self.dropout1 = nn.Dropout2d(0.25)
        self.dropout2 = nn.Dropout2d(0.5)
        self.fc1 = nn.Linear(9216, 128)
        self.fc2 = nn.Linear(128, 10) # Assuming 10 classes.  Adjust if needed for other datasets

    def forward(self, x):
        x = self.conv1(x)
        x = F.relu(x)
        x = self.conv2(x)
        x = F.relu(x)
        x = F.max_pool2d(x, 2)
        x = self.dropout1(x)
        x = torch.flatten(x, 1)
        x = self.fc1(x)
        x = F.relu(x)
        x = self.dropout2(x)
        x = self.fc2(x)
        output = F.log_softmax(x, dim=1)
        return output

if __name__ == '__main__':
    # Dataset and DataLoader
    ds_name = "{ds_name}"
    batch_size = 64
    if ds_name == "MNIST":
        input_channels = 1
        train_dataset = datasets.MNIST('./data', train=True, download=True,
                               transform=transforms.ToTensor())
        test_dataset = datasets.MNIST('./data', train=False, transform=transforms.ToTensor())
    elif ds_name == "CIFAR10":
        input_channels = 3
        train_dataset = datasets.CIFAR10('./data', train=True, download=True,
                                transform=transforms.Compose([
                                    transforms.ToTensor(),
                                    transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
                                ]))
        test_dataset = datasets.CIFAR10('./data', train=False, transform=transforms.Compose([
                                    transforms.ToTensor(),
                                    transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
                                ]))
    elif ds_name == "FashionMNIST":
        input_channels = 1
        train_dataset = datasets.FashionMNIST('./data', train=True, download=True,
                               transform=transforms.ToTensor())
        test_dataset = datasets.FashionMNIST('./data', train=False, transform=transforms.ToTensor())
    elif ds_name == "CIFAR100":
        input_channels = 3
        train_dataset = datasets.CIFAR100('./data', train=True, download=True,
                                transform=transforms.Compose([
                                    transforms.ToTensor(),
                                    transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
                                ]))
        test_dataset = datasets.CIFAR100('./data', train=False, transform=transforms.Compose([
                                    transforms.ToTensor(),
                                    transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
                                ]))
    else: # Default to MNIST-like grayscale if dataset is not recognized, assuming 1 channel. Adjust as needed.
        input_channels = 1
        train_dataset = datasets.MNIST('./data', train=True, download=True,
                               transform=transforms.ToTensor())
        test_dataset = datasets.MNIST('./data', train=False, transform=transforms.ToTensor())


    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

    # Model, Optimizer
    model = Net(input_channels=input_channels)
    optimizer = optim.Adam(model.parameters(), lr=1e-3)

    # Training loop
    epochs = 3 # Reduced epochs for example, increase for better training
    for epoch in range(epochs):
        model.train()
        for batch_idx, (data, target) in enumerate(train_loader):
            optimizer.zero_grad()
            output = model(data)
            loss = F.nll_loss(output, target)
            loss.backward()
            optimizer.step()
            if batch_idx % 100 == 0:
                print(f'Train Epoch: {epoch} [{batch_idx * len(data)}/{len(train_loader.dataset)} ({100. * batch_idx / len(train_loader):.0f}%)]\tLoss: {loss.item():.6f}')

        # Testing loop
        model.eval()
        test_loss = 0
        correct = 0
        with torch.no_grad():
            for data, target in test_loader:
                output = model(data)
                test_loss += F.nll_loss(output, target, reduction='sum').item()
                pred = output.argmax(dim=1, keepdim=True)
                correct += pred.eq(target.view_as(pred)).sum().item()

        test_loss /= len(test_loader.dataset)
        print(f'\nTest set: Average loss: {test_loss:.4f}, Accuracy: {correct}/{len(test_loader.dataset)} ({100. * correct / len(test_loader.dataset):.2f}%)\n')
