from torchvision import datasets
from llm_output import Net

train_ds = datasets.MNIST(root='./data', train=True, download=True)
test_ds = datasets.MNIST(root='./data', train=False, download=True)
print(train_ds.test_data.shape)

net = Net()

