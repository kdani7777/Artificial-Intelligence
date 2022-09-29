import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torchvision import datasets, transforms

# Feel free to import other packages, if needed.
# As long as they are supported by CSL machines.


def get_data_loader(training = True):
    """
    TODO: implement this function.

    INPUT:
        An optional boolean argument (default value is True for training dataset)

    RETURNS:
        Dataloader for the training set (if training = True) or the test set (if training = False)
    """
    custom_transform=transforms.Compose([
        transforms.ToTensor(),   # convert PIL image/numpy ndarray to tensor
        transforms.Normalize((0.1307,), (0.3081,))  # normalize tensor with specified mean and std
        ])

    # data for training our neural network
    train_set = datasets.MNIST('./data', train=True, download=True, transform=custom_transform)
    # data for model evaluation
    test_set = datasets.MNIST('./data', train=False, transform=custom_transform)

    train_loader = torch.utils.data.DataLoader(train_set, batch_size = 50)
    test_loader = torch.utils.data.DataLoader(test_set, batch_size = 50)

    if training:
        return train_loader
    else:
        return test_loader



def build_model():
    """
    TODO: implement this function.

    INPUT:
        None

    RETURNS:
        An untrained neural network model
    """
    model = nn.Sequential(
        nn.Flatten(),
        nn.Linear(784,128),
        nn.ReLU(),
        nn.Linear(128,64),
        nn.ReLU(),
        nn.Linear(64,10)
    )

    return model




def train_model(model, train_loader, criterion, T):
    """
    TODO: implement this function.

    INPUT:
        model - the model produced by the previous function
        train_loader  - the train DataLoader produced by the first function
        criterion   - cross-entropy
        T - number of epochs for training

    RETURNS:
        None
    """
    optimizer = optim.SGD(model.parameters(), lr=0.001, momentum=0.9)
    model.train() # put model in training mode

    for epoch in range(T): # loop over dataset for T epochs

        running_loss = 0.0
        total = 0
        correct = 0
        for i, data in enumerate(train_loader, 0):
            # get the inputs; data is list of [inputs, labels]
            inputs, labels = data

            # zero the parameter gradients
            optimizer.zero_grad()

            # forward + backward + optimize
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            # print statistics
            running_loss += loss.item()

            # count correct out of total
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()

        print('Train Epoch: %d Accuracy: %d/%d(%.2f%%) Loss: %.3f' %
         (epoch, correct, total,
         100 * correct / total,
         running_loss / len(train_loader)))



def evaluate_model(model, test_loader, criterion, show_loss = True):
    """
    TODO: implement this function.

    INPUT:
        model - the the trained model produced by the previous function
        test_loader    - the test DataLoader
        criterion   - cropy-entropy

    RETURNS:
        None
    """

    model.eval() # put model into evaluation mode

    running_loss = 0.0
    correct = 0
    total = 0
    with torch.no_grad():
        for data in test_loader:
            inputs, labels = data # inputs are the images
            outputs = model(inputs)

            # FIGURE OUT HOW TO CALCULATE LOSS
            loss = criterion(outputs, labels)
            running_loss += loss.item()

            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()

    if show_loss:
        print('Average loss: %.4f' % (running_loss / len(test_loader)))
        print('Accuracy: %.2f%%' % (100 * correct / total))
    else:
        print('Accuracy: %.2f%%' % (100 * correct / total))



def predict_label(model, test_images, index):
    """
    TODO: implement this function.

    INPUT:
        model - the trained model
        test_images   -  test image set of shape Nx1x28x28
        index   -  specific index  i of the image to be tested: 0 <= i <= N - 1


    RETURNS:
        None
    """

    class_names = ['zero', 'one', 'two', 'three', 'four', 'five', 'six', 'seven', 'eight', 'nine']

    # test_images is the input we're testing on
    # puts = model(test_images)
    output = model(test_images.dataset[index][0])

    prob = F.softmax(output, dim=1)

    """print(prob[0].data)
    print(len(prob[0].data))
    print(prob[0].data[0].item())"""

    top3 = sorted(zip(prob[0].data, class_names), reverse=True)[:3]

    print("%s: %.2f%%" % (top3[0][1], 100 * top3[0][0].item()))
    print("%s: %.2f%%" % (top3[1][1], 100 * top3[1][0].item()))
    print("%s: %.2f%%" % (top3[2][1], 100 * top3[2][0].item()))


if __name__ == '__main__':
    '''
    Feel free to write your own test code here to exaime the correctness of your functions.
    Note that this part will not be graded.
    '''
    criterion = nn.CrossEntropyLoss()
