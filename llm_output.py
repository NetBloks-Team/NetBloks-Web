import torch
import torch.nn as nn

class Net(nn.Module):
    def __init__(self, layer_params, input_channels=3, input_size=(32, 32)):
        super(Net, self).__init__()
        self.layer_list = []
        in_channels = input_channels
        current_in_size = list(input_size) # Make mutable
        flatten_output_size = None

        for layer_name, params in layer_params.items():
            layer_type = params['type']
            if layer_type == 'conv2d':
                filters = params['filters']
                kernel_size = params['kernel_size']
                activation = params['activation']
                self.layer_list.append(nn.Conv2d(in_channels, filters, kernel_size))
                if activation == 'relu':
                    self.layer_list.append(nn.ReLU())
                in_channels = filters
                current_in_size[0] = current_in_size[0] - kernel_size + 1
                current_in_size[1] = current_in_size[1] - kernel_size + 1
            elif layer_type == 'maxpool2d':
                pool_size = params['pool_size']
                self.layer_list.append(nn.MaxPool2d(pool_size))
                current_in_size[0] = current_in_size[0] // pool_size
                current_in_size[1] = current_in_size[1] // pool_size
            elif layer_type == 'flatten':
                self.layer_list.append(nn.Flatten())
                flatten_output_size = in_channels * current_in_size[0] * current_in_size[1]
            elif layer_type == 'dense':
                units = params['units']
                activation = params['activation']
                if flatten_output_size is None:
                    raise ValueError("Flatten layer must precede dense layer")
                self.layer_list.append(nn.Linear(flatten_output_size, units))
                if activation == 'relu':
                    self.layer_list.append(nn.ReLU())
            elif layer_type == 'dropout':
                rate = params['rate']
                self.layer_list.append(nn.Dropout(rate))
            else:
                raise ValueError(f"Unknown layer type: {layer_type}")

        self.layers = nn.Sequential(*self.layer_list)

    def forward(self, x):
        return self.layers(x)

if __name__ == '__main__':
    layer_params = {
        "layer-1": {"type":"conv2d", "filters": 32, "kernel_size": 3, "activation": "relu"},
        "layer-2": {"type":"conv2d", "filters": 64, "kernel_size": 3, "activation": "relu"},
        "layer-3": {"type":"maxpool2d", "pool_size": 2},
        "layer-4": {"type":"flatten"},
        "layer-5": {"type":"dense", "units": 128, "activation": "relu"},
        "layer-6": {"type":"dropout", "rate": 0.5}
    }
    net = Net(layer_params)
    print(net)
    input_tensor = torch.randn(1, 3, 32, 32)
    output = net(input_tensor)
    print(output.shape)
