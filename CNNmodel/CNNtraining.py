######################################################################################################
# You need to download torch, matplotlib, numpy, sklearn to run this file
######################################################################################################
#!pip install torch
#!pip install matplotlib
#!pip install numpy
#!pip install sklearn

#For this project we are initially using Kaggle notebook.
#importing packages
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset,DataLoader
import torchvision
from torchvision import datasets,transforms
import matplotlib.pyplot as plt
import numpy as np
from sklearn.model_selection import train_test_split

#Fix some training constants
batch_size = 8
learning_rate = 3e-4
num_epochs = 100
min_loss = 10000 #Defining this so we can save the least loss model
#Model saving path
PATH = './cnn.pth'

data_dir = '/kaggle/input/brain-tumor-classification-mri'
#prepare the dataloader
transform = transforms.Compose([transforms.ToTensor(),
                                transforms.Resize((132,132)),
                                transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])
train_data = datasets.ImageFolder(data_dir+'/Training/',transform = transform)
test_data = datasets.ImageFolder(data_dir+'/Testing/',transform = transform)
train_dataloader = torch.utils.data.DataLoader(train_data,batch_size = batch_size,shuffle = True)
test_dataloader = torch.utils.data.DataLoader(test_data,batch_size = batch_size,shuffle = True)


class ConvNet(nn.Module):
    def __init__(self, debug=False):
        super(ConvNet, self).__init__()
        # not using padding since the MRI picture, the tumor is always in the middle, thus we don't care about boundary of pictures
        self.dropout = nn.Dropout(0.3)
        self.conv1 = nn.Conv2d(3, 8, 5, stride=1)
        self.conv2 = nn.Conv2d(8, 32, 5, stride=1)
        self.conv3 = nn.Conv2d(32, 128, 3, stride=1, padding=1)
        self.pool = nn.MaxPool2d(2, 2)
        self.flat = nn.Flatten()
        self.fc1 = nn.Linear(128 * 15 * 15, 1024)
        self.fc2 = nn.Linear(1024, 128)
        self.fc3 = nn.Linear(128, 4)
        self.debug = debug

    def forward(self, x):
        x = F.relu(self.pool(self.conv1(x)))
        if self.debug:
            print(x.shape)
        x = F.relu(self.pool(self.conv2(x)))
        if self.debug:
            print(x.shape)
        x = F.relu(self.pool(self.conv3(x)))
        if self.debug:
            print(x.shape)
        x = self.flat(x)
        if self.debug:
            print(x.shape)
        x = F.relu(self.fc1(x))
        x = self.dropout(x)
        if self.debug:
            print(x.shape)
        x = F.relu(self.fc2(x))
        x = self.dropout(x)
        if self.debug:
            print(x.shape)
        x = F.relu(self.fc3(x))
        if self.debug:
            print(x.shape)
            exit()
        return x

    # We are using the kaggle GPU for free
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    model = ConvNet().to(device)
    # CrossEntropy loss is commonly used in multi classification problems
    criterion = nn.CrossEntropyLoss()

    # Use Adam optimizer(We tried a lot of optimizers but results are similar)
    optimizer = torch.optim.SGD(model.parameters(), lr=learning_rate, momentum=0.9)
    loss_record = []
    acc_record = []
    valid_acc_record = []
    # Training starts!
    for epoch in range(num_epochs):
        n_correct = 0
        for i, (images, labels) in enumerate(train_dataloader):
            # Push tensors to GPU
            images = images.to(device)
            labels = labels.to(device)

            # Forward pass
            outputs = model(images)
            loss = criterion(outputs, labels)

            # Backward and optimize
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

        # Save the model with least loss during epoches
        loss_record.append(loss.item())
        if loss.item() < min_loss:
            min_loss = loss.item()
            print("saving model")
            torch.save(model.state_dict(), PATH)
            # Inspect the loss for each epoch
        print(f'Epoch [{epoch + 1}/{num_epochs}], Loss: {loss.item():.4f}')

        # inspect the accuracy of validation after each training epoch
        with torch.no_grad():
            n_correct = 0
            n_samples = 0
            n_class_correct = [0, 0, 0, 0]
            n_class_samples = [0, 0, 0, 0]
            for images, labels in test_dataloader:
                images = images.to(device)
                labels = labels.to(device)
                # predict step
                outputs = model(images)
                # produce class given tensor
                _, predicted = torch.max(outputs, 1)
                n_samples += labels.size(0)
                n_correct += (predicted == labels).sum().item()

                for i in range(labels.shape[0]):
                    label = labels[i]
                    pred = predicted[i]
                    if (label == pred):
                        n_class_correct[label] += 1
                    n_class_samples[label] += 1

            acc = 100.0 * n_correct / n_samples
            print(f'Validation accuracy: {acc} %')
            valid_acc_record.append(acc)

    print("Finished!")
    epoches = [i + 1 for i in range(num_epochs)]
    plt.plot(epoches, loss_record)
    plt.show()
    plt.plot(epoches, valid_acc_record)
    plt.show()

    FILE = "CNN_model.pth"
    torch.save(model, FILE)

    # Testing the model
    testing_loss = []
    classes = ["glioma_tumor", "meningioma_tumor", "no_tomor", "pituitary_tomor"]
    model = ConvNet().to(device)
    model.load_state_dict(torch.load(PATH))
    model.to(device)
    with torch.no_grad():
        n_correct = 0
        n_samples = 0
        n_class_correct = [0, 0, 0, 0]
        n_class_samples = [0, 0, 0, 0]
        for images, labels in test_dataloader:
            images = images.to(device)
            labels = labels.to(device)
            # predict step
            outputs = model(images)
            # produce class given tensor
            _, predicted = torch.max(outputs, 1)
            n_samples += labels.size(0)
            n_correct += (predicted == labels).sum().item()

            for i in range(labels.shape[0]):
                label = labels[i]
                pred = predicted[i]
                if (label == pred):
                    n_class_correct[label] += 1
                n_class_samples[label] += 1

        acc = 100.0 * n_correct / n_samples
        print(f'Overall accuracy: {acc} %')

        for i in range(4):
            acc = 100.0 * n_class_correct[i] / n_class_samples[i]
            print(f'Accuracy of classifying {classes[i]}: {acc} %')