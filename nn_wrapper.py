import torch
from torch import nn as nn
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
import seaborn as sns
from matplotlib import pyplot as plt
from sklearn.metrics import confusion_matrix
from llm_output import Net

EPOCHS = 8

train_ds = datasets.MNIST(root='./data', train=True, transform=transforms.ToTensor(), download=True)
test_ds = datasets.MNIST(root='./data', train=False, transform=transforms.ToTensor(), download=True)
print("Train set size:", train_ds.data.shape)
print("Test set size:", test_ds.data.shape)
train_loader = DataLoader(
    train_ds, batch_size=64, shuffle=True, drop_last=True
)
test_loader = DataLoader(
    test_ds, batch_size=64, shuffle=False, drop_last=True
)

net = Net()
criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(net.parameters(), lr=0.01)
n_total_steps = len(train_loader)

for i in range(EPOCHS):
    # train
    running_loss = 0.0  # this keeps track of the loss per epoch
    # print("Epoch:", i, "start...")
    for b, (X_train, y_train) in enumerate(train_loader):
        X_train, y_train = X_train.to("cpu"), y_train.to("cpu")
        # forward pass
        y_pred = net.forward(X_train)
        loss = criterion(y_pred, y_train)
        # backward pass
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        # record loss
        running_loss += loss.item()
    print("Epoch:", i, "of", EPOCHS, "   loss:", running_loss / n_total_steps)

print("Done with ANN fitting.")

X_test = torch.reshape(test_ds.data.float(), (-1, 1, 28, 28))
y_test = test_ds.targets
print(X_test.shape)
with torch.no_grad():
    y_pred = net.forward(X_test)

confusion_mtx = confusion_matrix(y_test, y_pred)
hmap = sns.heatmap(
   confusion_mtx, annot=True, fmt="g"
)  # Create the data visualization
plt.show()
