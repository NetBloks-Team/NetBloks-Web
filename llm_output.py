import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torchvision import datasets, transforms
from torch.utils.data import DataLoader

class Net(nn.Module):
    def __init__(self, layer_config):
        super(Net, self).__init__()
        self.layers = nn.ModuleList()
        in_channels = 1  # MNIST images are grayscale, so 1 input channel
        in_features = None # To keep track of input features for linear layers

        for layer_params in layer_config:
            layer_type = layer_params['type']

            if layer_type == 'Conv2d':
                channels = int(layer_params['channels'])
                kernel_size = int(layer_params['kernel_size'])
                stride = int(layer_params['stride']) if layer_params['stride'] else 1
                self.layers.append(nn.Conv2d(in_channels, channels, kernel_size, stride=stride))
                in_channels = channels # Update in_channels for the next layer

            elif layer_type == 'ReLU':
                self.layers.append(nn.ReLU())

            elif layer_type == 'MaxPool2d':
                kernel_size = int(layer_params['kernel_size'])
                stride = int(layer_params['stride']) if layer_params['stride'] else None # Use None for default stride (kernel_size)
                self.layers.append(nn.MaxPool2d(kernel_size, stride=stride))

            elif layer_type == 'Dropout':
                p = float(layer_params['p'])
                self.layers.append(nn.Dropout(p))

            elif layer_type == 'Linear':
                out_features = int(layer_params['out_features'])
                if in_features is None:
                    # Placeholder, will be determined after convolutional layers in forward pass
                    in_features = -1
                self.layers.append(nn.Linear(in_features, out_features))
                in_features = out_features # Update in_features for the next layer


        self.layer_config = layer_config

    def forward(self, x):
        # Placeholder to determine the in_features for the first linear layer
        first_linear_layer_index = -1
        for i, layer_params in enumerate(self.layer_config):
            if layer_params['type'] == 'Linear':
                first_linear_layer_index = i
                break

        if first_linear_layer_index != -1:
            temp_x = x
            for i in range(first_linear_layer_index):
                layer = self.layers[i]
                temp_x = layer(temp_x)

            if self.layers[first_linear_layer_index].in_features == -1:
                # Calculate in_features for the first linear layer dynamically
                in_features = temp_x.view(temp_x.size(0), -1).size(1)
                self.layers[first_linear_layer_index] = nn.Linear(in_features, int(self.layer_config[first_linear_layer_index]['out_features']))
                # Re-initialize weights and biases for the newly created linear layer.
                # Otherwise it will have random initializations which might not be desired.
                nn.init.xavier_uniform_(self.layers[first_linear_layer_index].weight)
                nn.init.zeros_(self.layers[first_linear_layer_index].bias)


        x = x # Start from original input again for the actual forward pass
        for layer in self.layers:
            if isinstance(layer, nn.MaxPool2d):
                x = layer(x)
            elif isinstance(layer, nn.ReLU):
                x = layer(x)
            elif isinstance(layer, nn.Dropout):
                x = layer(x)
            elif isinstance(layer, nn.Conv2d):
                x = layer(x)
            elif isinstance(layer, nn.Linear):
                x = x.view(x.size(0), -1) # Flatten before linear layer
                x = layer(x)
            else:
                raise TypeError("Layer type not supported")
        return F.log_softmax(x, dim=1) # Use log_softmax for NLLLoss


def train(model, device, train_loader, optimizer, epoch):
    model.train()
    for batch_idx, (data, target) in enumerate(train_loader):
        data, target = data.to(device), target.to(device)
        optimizer.zero_grad()
        output = model(data)
        loss = F.nll_loss(output, target) # Use NLLLoss for log_softmax output
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
            test_loss += F.nll_loss(output, target, reduction='sum').item() # sum up batch loss
            pred = output.argmax(dim=1, keepdim=True) # get the index of the max log-probability
            correct += pred.eq(target.view_as(pred)).sum().item()

    test_loss /= len(test_loader.dataset)

    print('\nTest set: Average loss: {:.4f}, Accuracy: {}/{} ({:.0f}%)\n'.format(
        test_loss, correct, len(test_loader.dataset),
        100. * correct / len(test_loader.dataset)))


def main():
    # Define the network parameters from the problem description
    layer_config = [
        {'type': 'Conv2d', 'channels': '6', 'kernel_size': '3', 'stride': ''},
        {'type': 'ReLU'},
        {'type': 'MaxPool2d', 'kernel_size': '2', 'stride': ''},
        {'type': 'Conv2d', 'channels': '16', 'kernel_size': '3', 'stride': ''},
        {'type': 'ReLU'},
        {'type': 'MaxPool2d', 'kernel_size': '2', 'stride': ''},
        {'type': 'Dropout', 'p': '0.25'},
        {'type': 'Linear', 'out_features': '84'},
        {'type': 'ReLU'},
        {'type': 'Linear', 'out_features': '10'}
    ]

    # Training settings
    batch_size = 64
    test_batch_size = 1000
    epochs = 10
    learning_rate = 0.01
    momentum = 0.5
    no_cuda = False
    seed = 1
    log_interval = 10

    use_cuda = not no_cuda and torch.cuda.is_available()

    torch.manual_seed(seed)

    device = torch.device("cuda" if use_cuda else "cpu")

    train_kwargs = {'batch_size': batch_size}
    test_kwargs = {'batch_size': test_batch_size}
    if use_cuda:
        cuda_kwargs = {'num_workers': 1,
                       'pin_memory': True,
                       'shuffle': True}
        train_kwargs.update(cuda_kwargs)
        test_kwargs.update(cuda_kwargs)

    transform=transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.1307,), (0.3081,))
        ])
    dataset1 = datasets.MNIST('../data', train=True, download=True,
                       transform=transform)
    dataset2 = datasets.MNIST('../data', train=False,
                       transform=transform)
    train_loader = DataLoader(dataset1,**train_kwargs)
    test_loader = DataLoader(dataset2, **test_kwargs)

    model = Net(layer_config).to(device)
    optimizer = optim.SGD(model.parameters(), lr=learning_rate, momentum=momentum)

    for epoch in range(1, epochs + 1):
        train(model, device, train_loader, optimizer, epoch)
        test(model, device, test_loader)

    # Save the trained model (optional)
    # torch.save(model.state_dict(), "mnist_cnn.pt")


if __name__ == '__main__':
    main()
