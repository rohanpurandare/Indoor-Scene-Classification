import torch
import torch.nn as nn
import torch.nn.functional as F
from sklearn.neighbors import NearestNeighbors
import numpy as np
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
import PIL.Image
from data_loader import create_data_loaders, get_label_to_idx
from sklearn.metrics import accuracy_score, classification_report
from sklearn.decomposition import PCA
from torch.nn.parallel import DataParallel
from torchvision import models
import torch.nn.functional as F
import torch.nn as nn
from sklearn.model_selection import KFold
import matplotlib.pyplot as plt

if __name__ == '__main__':
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f'Using device: {device}')

    '''
    Models we are using:
    1. KNN: K-Nearest Neighbors
    2. SVM: Support Vector Machine
    3. CNN: Convolutional Neural Network
    '''


    class CNN(nn.Module):
        def __init__(self, num_classes=len(get_label_to_idx())):
            super(CNN, self).__init__()
            self.features = nn.Sequential(
                nn.Conv2d(3, 64, kernel_size=3, padding=1),
                nn.ReLU(inplace=True),
                nn.MaxPool2d(kernel_size=2, stride=2),

                nn.Conv2d(64, 128, kernel_size=3, padding=1),
                nn.ReLU(inplace=True),
                nn.MaxPool2d(kernel_size=2, stride=2),

                nn.Conv2d(128, 256, kernel_size=3, padding=1),
                nn.ReLU(inplace=True),
                nn.MaxPool2d(kernel_size=2, stride=2),

                nn.Conv2d(256, 256, kernel_size=3, padding=1),
                nn.ReLU(inplace=True),
                nn.MaxPool2d(kernel_size=2, stride=2)
            )
            self.classifier = nn.Sequential(
                nn.Dropout(),
                nn.Linear(256 * 18 * 18, 128),
                nn.ReLU(inplace=True),
                nn.Dropout(),
                nn.Linear(128, num_classes)
            )

        def forward(self, x):
            x = self.features(x)
            x = torch.flatten(x, 1)
            x = self.classifier(x)
            return x

    class ResNet(nn.Module):
        def __init__(self, num_classes=len(get_label_to_idx())):
            super(ResNet, self).__init__()
            self.resnet = models.resnet18(weights="DEFAULT")
            self.resnet.fc = nn.Linear(512, num_classes)

        def forward(self, x):
            return torch.sigmoid(self.resnet(x))

    class Inception(nn.Module):
        def __init__(self, num_classes=len(get_label_to_idx())):
            super(Inception, self).__init__()
            self.inception = models.inception_v3(weights="DEFAULT")
            self.inception.fc = nn.Linear(2048, num_classes)
            self.name = "inception"
        def forward(self, x):
            output = self.inception(x)
            return torch.sigmoid(output.logits)

    train, test = create_data_loaders(sampling='random')
    print(f'Training on {len(train)} batches')
    #train.dataset.images = torch.transpose(train.dataset.images, 1, 3).float()
    #print(train.dataset.images.shape)
    model = Inception()
    model = model.to(device)
    print("Number of parameters: ", sum(p.numel() for p in model.parameters() if p.requires_grad))
    optimizer = torch.optim.Adam(model.parameters(), lr=6e-5)
    epochs = 11

    dataset_indices = np.arange(len(train.dataset))
    kf = KFold(n_splits=5)
    folds = []
    for train_indices, val_indices in kf.split(dataset_indices):
        folds.append((train_indices, val_indices))

    train_losses = np.zeros((5, epochs))
    val_losses = np.zeros((5, epochs))

    train_accuracies = []
    validation_accuracies = []

    for fold, (train_indices, val_indices) in enumerate(folds):
        print(f'Fold {fold+1}')
        model = Inception()
        model.to(device)
        optimizer = torch.optim.Adam(model.parameters(), lr=6e-5)
        train_loader = torch.utils.data.DataLoader(train.dataset, batch_size=64, sampler=torch.utils.data.SubsetRandomSampler(train_indices))
        val_loader = torch.utils.data.DataLoader(train.dataset, batch_size=64, sampler=torch.utils.data.SubsetRandomSampler(val_indices))
        model.train()
        for epoch in range(epochs):
            epoch_loss = 0
            for i, (images, labels) in enumerate(train_loader):
                images, labels = images.to(device), labels.to(device)
                optimizer.zero_grad()
                outputs = model(images)
                loss = F.cross_entropy(outputs, labels)
                loss.backward()
                optimizer.step()
                epoch_loss += loss.item()

            epoch_loss /= len(train_loader)
            train_losses[fold, epoch] = epoch_loss
            print(f'Epoch {epoch+1} Loss: {epoch_loss}')

        model.eval()
        with torch.no_grad():
            correct = 0
            total = 0
            loss_val = 0

            for images, labels in train_loader:
                images, labels = images.to(device), labels.to(device)
                outputs = model(images)
                loss = F.cross_entropy(outputs, labels)
                loss_val += loss.item()
                _, predicted = torch.max(outputs.data, 1)
                total += labels.size(0)
                correct += (predicted == labels).sum().item()

            train_accuracies.append(100 * correct / total)
            print(f'Accuracy of the network on the training images: {100 * correct / total} %')
            correct = 0
            total = 0

            for images, labels in val_loader:
                images, labels = images.to(device), labels.to(device)
                outputs = model(images)
                loss = F.cross_entropy(outputs, labels)
                loss_val += loss.item()
                _, predicted = torch.max(outputs.data, 1)
                total += labels.size(0)
                correct += (predicted == labels).sum().item()

            validation_accuracies.append(100 * correct / total)
            loss_val /= len(val_loader)
            val_losses[fold, epoch] = loss_val
            print(f'Validation Loss: {loss_val}')
            print(f'Accuracy of the network on the validation images: {100 * correct / total} %')

            with torch.no_grad():
                correct = 0
                total = 0
                for images, labels in test:
                    images, labels = images.to(device), labels.to(device)
                    outputs = model(images)
                    _, predicted = torch.max(outputs.data, 1)
                    total += labels.size(0)
                    correct += (predicted == labels).sum().item()

                print(f'Accuracy of the network on the test images: {100 * correct / total} %')

    best_fold = np.argmin(val_losses[:, -1])
    print(f'Best fold: {best_fold}')
    plt.figure(figsize=(10, 6))
    plt.plot(np.arange(epochs), train_losses[best_fold], label='Training Loss')
    plt.plot(np.arange(epochs), val_losses[best_fold], label='Validation Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.legend()
    plt.savefig('loss_plot_res_net_18_pretrained.png')