import torch
from torch import nn as nn
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
from llm_output import Net


EPOCHS = 8

def run_model(ds_name: str, printer = None) -> float:
    if ds_name == "MNIST":
        train_ds = datasets.MNIST(root='./data', train=True, transform=transforms.ToTensor(), download=True)
        test_ds = datasets.MNIST(root='./data', train=False, transform=transforms.ToTensor(), download=True)
    elif ds_name == "CIFAR 10":
        train_ds = datasets.CIFAR10(root='./data', train=True, transform=transforms.ToTensor(), download=True)
        test_ds = datasets.CIFAR10(root='./data', train=False, transform=transforms.ToTensor(), download=True)
    elif ds_name == "CIFAR 100":
        train_ds = datasets.CIFAR100(root='./data', train=True, transform=transforms.ToTensor(), download=True)
        test_ds = datasets.CIFAR100(root='./data', train=False, transform=transforms.ToTensor(), download=True)
    else:
        raise ValueError(f"Unknown dataset: {ds_name}")
    printer(f"Training on {train_ds.data.shape[0]} data points.") # Send this info to the user terminal
    printer(f"Testing on {test_ds.data.shape[0]} data points.")
    train_loader = DataLoader(
        train_ds, batch_size=64, shuffle=True, drop_last=True
    )

    net = Net()
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(net.parameters(), lr=0.01)
    n_total_steps = len(train_loader)

    prev_model_loss = None  # this keeps track of the loss from the previous epoch
    low_flag = False
    high_flag = False

    for i in range(EPOCHS):
        # train
        running_loss = 0.0  # this keeps track of the loss per epoch
        # print("Epoch:", i, "start...")
        for _, (X_train, y_train) in enumerate(train_loader):
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
        model_loss = running_loss / n_total_steps
        printer(f"Training cycle: {i} of {EPOCHS}. Your current loss is {model_loss}.\nLower loss means your model is performing better.")
        if prev_model_loss is not None:
            if model_loss < prev_model_loss and not low_flag:
                printer("Your model loss has decreased. This means your model is performing well.")
                low_flag = True
            elif model_loss > prev_model_loss and not high_flag:
                printer("Your model loss has increased. This could mean your model is exploring the data space. If this continues, your model may be overfitting.")
                high_flag = True
        prev_model_loss = model_loss

    printer("Network training done!")

    X_test = torch.reshape(test_ds.data.float(), (-1, 1, 28, 28))
    y_test = test_ds.targets
    with torch.no_grad():
        y_pred = net.forward(X_test)
    accuracy = 0
    for i in range (len(y_pred)):
        print(y_pred[i])
        if y_pred[i] == y_test[i]:
            accuracy += 1
    accuracy = accuracy / len(y_pred)
    printer(f"Your model performed with an accuracy of: {accuracy}%")

    # confusion_mtx = confusion_matrix(y_test, y_pred)
    # hmap = sns.heatmap(
    #    confusion_mtx, annot=True, fmt="g"
    # )  # Create the data visualization
    # plt.show()

if __name__ == "__main__":
    # Example usage
    run_model("MNIST", printer=print)