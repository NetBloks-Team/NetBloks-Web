import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision
import torchvision.transforms as transforms
import torch.optim as optim

class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        self.layers = nn.ModuleList()
        layer_config = [{'type': 'Conv2d', 'out_channels': '16', 'kernel_size': '3', 'stride': ''}, {'type': 'ReLU'}, {'type': 'MaxPool2d', 'kernel_size': '2', 'stride': ''}, {'type': 'Conv2d', 'out_channels': '64', 'kernel_size': '3', 'stride': ''}, {'type': 'ReLU'}, {'type': 'MaxPool2d', 'kernel_size': '2', 'stride': ''}, {'type': 'Dropout', 'p': '0.25'}, {'type': 'Linear', 'out_features': '840'}, {'type': 'ReLU'}, {'type': 'Linear', 'out_features': '100'}]
        in_channels = 1
        for layer_param in layer_config:
            layer_type = layer_param['type']
            if layer_type == 'Conv2d':
                out_channels = int(layer_param['out_channels'])
                kernel_size = int(layer_param['kernel_size'])
                stride = int(layer_param['stride']) if layer_param['stride'] != '' else 1
                self.layers.append(nn.Conv2d(in_channels, out_channels, kernel_size, stride=stride))
                in_channels = out_channels
            elif layer_type == 'ReLU':
                self.layers.append(nn.ReLU())
            elif layer_type == 'MaxPool2d':
                kernel_size = int(layer_param['kernel_size'])
                stride = int(layer_param['stride']) if layer_param['stride'] != '' else None
                self.layers.append(nn.MaxPool2d(kernel_size, stride=stride))
            elif layer_type == 'Dropout':
                p = float(layer_param['p'])
                self.layers.append(nn.Dropout(p))
            elif layer_type == 'Linear':
                out_features = int(layer_param['out_features'])
                if layer_param == layer_config[7]:
                    self.layers.append(nn.Linear(64 * 5 * 5, out_features))
                else:
                    prev_layer_out_features = int(layer_config[layer_config.index(layer_param)-2]['out_features'])
                    self.layers.append(nn.Linear(prev_layer_out_features, out_features))
        self.flatten = nn.Flatten()

    def forward(self, x):
        for layer in self.layers[:7]:
            x = layer(x)
        x = self.flatten(x)
        for layer in self.layers[7:]:
            x = layer(x)
        return x

transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.1307,), (0.3081,))
])

train_dataset = torchvision.datasets.MNIST(root='./data', train=True, download=True, transform=transform)
test_dataset = torchvision.datasets.MNIST(root='./data', train=False, download=True, transform=transform)

train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=64, shuffle=True)
test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=1000, shuffle=False)

model = Net()
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters())

epochs = 2
for epoch in range(epochs):
    model.train()
    for batch_idx, (data, target) in enumerate(train_loader):
        optimizer.zero_grad()
        output = model(data)
        loss = criterion(output, target)
        loss.backward()
        optimizer.step()
        if batch_idx % 100 == 0:
            print(f'Train Epoch: {epoch} [{batch_idx * len(data)}/{len(train_loader.dataset)} ({100. * batch_idx / len(train_loader):.0f}%)]\tLoss: {loss.item():.6f}')

    model.eval()
    test_loss = 0
    correct = 0
    with torch.no_grad():
        for data, target in test_loader:
            output = model(data)
            test_loss += criterion(output, target).item()
            pred = output.argmax(dim=1, keepdim=True)
            correct += pred.eq(target.view_as(pred)).sum().item()

    test_loss /= len(test_loader)
    print(f'\nTest set: Average loss: {test_loss:.4f}, Accuracy: {correct}/{len(test_loader.dataset)} ({100. * correct / len(test_loader.dataset):.0f}%)\n')
