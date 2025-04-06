import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torchvision import datasets, transforms

class Flatten(nn.Module):
    def forward(self, x):
        return x.view(x.size(0), -1)

class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        self.layers_config = [{'type': 'Conv2d', 'channels': '32', 'kernel_size': '3', 'stride': ''}, {'type': 'ReLU'}, {'type': 'Conv2d', 'channels': '64', 'kernel_size': '3', 'stride': ''}, {'type': 'ReLU'}, {'type': 'MaxPool2d', 'kernel_size': '2', 'stride': ''}, {'type': 'Flatten'}, {'type': 'Dropout', 'p': '0.3'}, {'type': 'Linear', 'out_features': '10'}]
        self.layers = nn.ModuleList()
        in_channels = 1
        for layer_config in self.layers_config:
            layer_type = layer_config['type']
            if layer_type == 'Conv2d':
                channels = int(layer_config['channels'])
                kernel_size = int(layer_config['kernel_size'])
                stride = int(layer_config['stride']) if layer_config['stride'] else 1
                self.layers.append(nn.Conv2d(in_channels, channels, kernel_size, stride=stride))
                in_channels = channels
            elif layer_type == 'ReLU':
                self.layers.append(nn.ReLU())
            elif layer_type == 'MaxPool2d':
                kernel_size = int(layer_config['kernel_size'])
                stride = int(layer_config['stride']) if layer_config['stride'] else kernel_size
                self.layers.append(nn.MaxPool2d(kernel_size, stride=stride))
            elif layer_type == 'Flatten':
                self.layers.append(Flatten())
            elif layer_type == 'Dropout':
                p = float(layer_config['p'])
                self.layers.append(nn.Dropout(p))
            elif layer_type == 'Linear':
                out_features = int(layer_config['out_features'])
                self.layers.append(nn.Linear(self._calculate_linear_input_size(), out_features))

    def _calculate_linear_input_size(self):
        dummy_input = torch.randn(1, 1, 28, 28)
        x = dummy_input
        for layer_config, layer in zip(self.layers_config[:-1], self.layers[:-1]): # Exclude the final Linear layer
            layer_type = layer_config['type']
            if layer_type == 'Conv2d':
                x = layer(x)
            elif layer_type == 'ReLU':
                x = layer(x)
            elif layer_type == 'MaxPool2d':
                x = layer(x)
            elif layer_type == 'Flatten':
                x = layer(x)
            elif layer_type == 'Dropout':
                x = layer(x)
        return x.view(1, -1).size(1)


    def forward(self, x):
        for layer_config, layer in zip(self.layers_config, self.layers):
            layer_type = layer_config['type']
            if layer_type == 'Conv2d':
                x = F.relu(layer(x))
            elif layer_type == 'ReLU':
                pass # ReLU is applied in Conv2d
            elif layer_type == 'MaxPool2d':
                x = layer(x)
            elif layer_type == 'Flatten':
                x = layer(x)
            elif layer_type == 'Dropout':
                x = layer(x)
            elif layer_type == 'Linear':
                x = layer(x)
        return F.log_softmax(x, dim=1)

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

def main():
    # Training settings
    batch_size = 64
    test_batch_size = 1000
    epochs = 10
    lr = 0.01
    momentum = 0.5
    no_cuda = False
    seed = 1
    log_interval = 10

    use_cuda = not no_cuda and torch.cuda.is_available()

    torch.manual_seed(seed)

    device = torch.device("cuda" if use_cuda else "cpu")

    kwargs = {'num_workers': 1, 'pin_memory': True} if use_cuda else {}
    train_loader = torch.utils.data.DataLoader(
        datasets.MNIST('../data', train=True, download=True,
                       transform=transforms.Compose([
                           transforms.ToTensor(),
                           transforms.Normalize((0.1307,), (0.3081,))
                       ])),
        batch_size=batch_size, shuffle=True, **kwargs)
    test_loader = torch.utils.data.DataLoader(
        datasets.MNIST('../data', train=False, transform=transforms.Compose([
                           transforms.ToTensor(),
                           transforms.Normalize((0.1307,), (0.3081,))
                       ])),
        batch_size=test_batch_size, shuffle=True, **kwargs)

    model = Net().to(device)
    optimizer = optim.SGD(model.parameters(), lr=lr, momentum=momentum)

    for epoch in range(1, epochs + 1):
        train(model, device, train_loader, optimizer, epoch)
        test(model, device, test_loader)

    torch.save(model.state_dict(), "mnist_cnn.pt")


if __name__ == '__main__':
    main()
