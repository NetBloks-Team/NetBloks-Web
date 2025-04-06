import time
import importlib
import torch
from torch import nn as nn
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
from torchtext import datasets as tdatasets
import llm_output
import gemini_gen

def run_model(ds_name: str, printer = None, nn_struct: str = None, epochs: int = 8) -> float:
    printer("Loading dataset...")
    if ds_name == "MNIST":
        train_ds = datasets.MNIST(root='./data', train=True, transform=transforms.ToTensor(), download=True)
        test_ds = datasets.MNIST(root='./data', train=False, transform=transforms.ToTensor(), download=True)
    elif ds_name == "CIFAR 10":
        train_ds = datasets.CIFAR10(root='./data', train=True, transform=transforms.ToTensor(), download=True)
        test_ds = datasets.CIFAR10(root='./data', train=False, transform=transforms.ToTensor(), download=True)
    elif ds_name == "CIFAR 100":
        train_ds = datasets.CIFAR100(root='./data', train=True, transform=transforms.ToTensor(), download=True)
        test_ds = datasets.CIFAR100(root='./data', train=False, transform=transforms.ToTensor(), download=True)
    elif ds_name == "KMNIST":
        train_ds = datasets.KMNIST(root='./data', train=True, transform=transforms.ToTensor(), download=True)
        test_ds = datasets.KMNIST(root='./data', train=False, transform=transforms.ToTensor(), download=True)
    elif ds_name == "SVHN":
        train_ds = datasets.SVHN(root='./data', split='train', transform=transforms.ToTensor(), download=True)
        test_ds = datasets.SVHN(root='./data', split='test', transform=transforms.ToTensor(), download=True)
    else:
        raise ValueError(f"Unknown dataset: {ds_name}")
    printer(f"Training on {train_ds.data.shape[0]} data points.")
    train_loader = DataLoader(
        train_ds, batch_size=64, shuffle=True, drop_last=True
    )

    net = llm_output.Net()
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(net.parameters(), lr=0.01)
    n_total_steps = len(train_loader)

    prev_model_loss = None  # this keeps track of the loss from the previous epoch
    low_flag = False
    high_flag = False

    for i in range(epochs):
        # train
        running_loss = 0.0  # this keeps track of the loss per epoch
        # print("Epoch:", i, "start...")
        for _, (X_train, y_train) in enumerate(train_loader):
            X_train, y_train = X_train.to("cpu"), y_train.to("cpu")
            # forward pass
            try:
                y_pred = net.forward(X_train)
            except Exception as e:
                gemini_gen.gemini_gen(ds_name, nn_struct, str(e))
                printer(f"An error occurred in the original neural network, regenerating")
                importlib.reload(llm_output)
                net = llm_output.Net()
                try:
                    y_pred = net.forward(X_train)
                except Exception as e:
                    printer(f"An error occurred in the regenerated neural network, terminating in 5 seconds.")
                    time.sleep(5)
                    raise e
            loss = criterion(y_pred, y_train)
            # backward pass
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            # record loss
            running_loss += loss.item()
        model_loss = running_loss / n_total_steps
        printer(f"Training cycle: {i+1} of {epochs}. Your current loss is {model_loss}.")
        if prev_model_loss is not None:
            if model_loss < prev_model_loss and not low_flag:
                printer("Your model loss has decreased. This means your model is performing well.")
                low_flag = True
            elif model_loss > prev_model_loss and not high_flag:
                printer("Your model loss has increased. This could mean your model is exploring the data space. If this continues, your model may be overfitting.")
                high_flag = True
        prev_model_loss = model_loss

    printer("Network training done!")

    printer(f"Testing on {test_ds.data.shape[0]} data points.")

    X_test = torch.reshape(test_ds.data.float(), (-1, 1, 28, 28))
    y_test = test_ds.targets
    with torch.no_grad():
        y_pred = net.forward(X_test)
    accuracy = 0
    for i in range (len(y_pred)):
        if torch.argmax(y_pred[i]) == y_test[i]:
            accuracy += 1
    accuracy = accuracy / len(y_pred)
    printer(f"Your model performed with an accuracy of: {accuracy*100}%")

    # confusion_mtx = confusion_matrix(y_test, y_pred)
    # hmap = sns.heatmap(
    #    confusion_mtx, annot=True, fmt="g"
    # )  # Create the data visualization
    # plt.show()

if __name__ == "__main__":
    # Example usage
    run_model("MNIST", printer=print)