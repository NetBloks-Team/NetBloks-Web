import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torchvision import datasets, transforms
from torch.utils.data import DataLoader

class Net(nn.Module):
    def __init__(self, input_channels):
        super(Net, self).__init__()
        # Example layers - please replace with your {nn_struct} parameters
        self.conv1 = nn.Conv2d(input_channels, 32, 3, 1)
        self.conv2 = nn.Conv2d(32, 64, 3, 1)
        self.dropout1 = nn.Dropout2d(0.25)
        self.dropout2 = nn.Dropout2d(0.5)
        self.fc1 = nn.Linear(9216, 128) # Example, adjust based on conv output and {nn_struct}
        self.fc2 = nn.Linear(128, 10) # Assuming 10 classes for classification, adjust if needed

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
    # Dataset and training parameters - please replace {ds_name} and adjust as needed
    ds_name = "{ds_name}" # e.g., 'MNIST' or 'CIFAR10'
    batch_size = 64
    learning_rate = 0.01
    epochs = 10
    use_cuda = torch.cuda.is_available()
    device = torch.device("cuda" if use_cuda else "cpu")
    kwargs = {'num_workers': 1, 'pin_memory': True} if use_cuda else {}

    # Determine input channels based on dataset name
    if ds_name.upper() == 'MNIST':
        input_channels = 1
        dataset_class = datasets.MNIST
    elif ds_name.upper() == 'CIFAR10':
        input_channels = 3
        dataset_class = datasets.CIFAR10
    else:
        input_channels = 3 # Default to 3 channels if dataset is not recognized, adjust if needed
        dataset_class = None # You will need to define your dataset loading if it's not MNIST or CIFAR10
        print(f"Warning: Dataset '{ds_name}' not recognized. Defaulting to 3 input channels. Please adjust input_channels and dataset loading if needed.")

    if dataset_class is not None: # Proceed only if dataset class is recognized (MNIST or CIFAR10 for example)
        # Data transformations - adjust as needed for your dataset
        transform=transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize((0.1307,), (0.3081,)) # Example normalization for MNIST, adjust for other datasets
        ])
        dataset1 = dataset_class('../data', train=True, download=True, transform=transform)
        dataset2 = dataset_class('../data', train=False, transform=transform)
        train_loader = DataLoader(dataset1, batch_size=batch_size, shuffle=True, **kwargs)
        test_loader = DataLoader(dataset2, batch_size=1000, shuffle=True, **kwargs)

        # Initialize the network
        model = Net(input_channels).to(device)
        optimizer = optim.Adam(model.parameters(), lr=learning_rate) # Example optimizer, adjust as needed

        def train(model, device, train_loader, optimizer, epoch):
            model.train()
            for batch_idx, (data, target) in enumerate(train_loader):
                data, target = data.to(device), target.to(device)
                optimizer.zero_grad()
                output = model(data)
                loss = F.nll_loss(output, target)
                loss.backward()
                optimizer.step()
                if batch_idx % 100 == 0:
                    print('Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}'.format(
                        epoch, batch_idx * len(data), len(train_loader.dataset),
                        100. * batch_idx / len(train_loader), loss.item()))

        def test(model, device, test_loader):
            model.eval()
            test_loss = 0
            correct = 0
            with torch.no_grad():
                for data, target in test_loader:
                    data, target = data.to(device), target.to(device)
                    output = model(data)
                    test_loss += F.nll_loss(output, target, reduction='sum').item()  # sum up batch loss
                    pred = output.argmax(dim=1, keepdim=True)  # get the index of the max log-probability
                    correct += pred.eq(target.view_as(pred)).sum().item()

            test_loss /= len(test_loader.dataset)

            print('\nTest set: Average loss: {:.4f}, Accuracy: {}/{} ({:.0f}%)\n'.format(
                test_loss, correct, len(test_loader.dataset),
                100. * correct / len(test_loader.dataset)))

        for epoch in range(1, epochs + 1):
            train(model, device, train_loader, optimizer, epoch)
            test(model, device, test_loader)
    else:
        print("Dataset loading not implemented for the specified dataset. Please check dataset name and implement data loading if needed.")
