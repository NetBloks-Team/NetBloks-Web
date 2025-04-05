import torch
import torch.nn as nn

class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        self.layer1_conv2d = nn.Conv2d(3, 32, kernel_size=3)
        self.layer1_relu = nn.ReLU()
        self.layer2_conv2d = nn.Conv2d(32, 64, kernel_size=3)
        self.layer2_relu = nn.ReLU()
        self.layer3_maxpool2d = nn.MaxPool2d(kernel_size=2)
        self.layer4_flatten = nn.Flatten()
        self.layer5_dense = nn.Linear(64 * 14 * 14, 128) # Placeholder, will be adjusted in forward if needed
        self.layer5_relu = nn.ReLU()
        self.layer6_dropout = nn.Dropout(0.5)

    def forward(self, x):
        x = self.layer1_conv2d(x)
        x = self.layer1_relu(x)
        x = self.layer2_conv2d(x)
        x = self.layer2_relu(x)
        x = self.layer3_maxpool2d(x)
        x = self.layer4_flatten(x)
        if self.layer5_dense.in_features != x.shape[1]:
            self.layer5_dense = nn.Linear(x.shape[1], 128) # Adjust dense layer input size dynamically
            self.layer5_dense.to(x.device) # Move to the same device if needed
        x = self.layer5_dense(x)
        x = self.layer5_relu(x)
        x = self.layer6_dropout(x)
        return x

if __name__ == '__main__':
    net = Net()
    print(net)

    # Example input
    input_tensor = torch.randn(1, 3, 32, 32) # Batch size 1, 3 channels, 32x32 image
    output = net(input_tensor)
    print(output.shape) # Expected output shape: torch.Size([1, 128])
