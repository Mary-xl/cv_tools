# A practice for training classifier

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
import torch.utils.data.dataset as Dataset
import torchvision
import torchvision.transforms as transforms
import torch.optim as optim
import matplotlib.pyplot as plt

class Net(nn.Module):
    def __init__(self):
        super(Net,self).__init__()
        self.conv1=nn.Conv2d(3,6,5)
        self.relu1=nn.ReLU(inplace=True)
        self.pool1=nn.MaxPool2d(kernel_size=2)
        self.conv2=nn.Conv2d(6,16,5)
        self.relu2=nn.ReLU(inplace=True)
        self.pool2=nn.MaxPool2d(kernel_size=2)
        self.fc1=nn.Linear(16*5*5,120)
        self.relu3=nn.ReLU(inplace=True)
        self.fc2=nn.Linear(120,84)
        self.relu4=nn.ReLU(inplace=True)
        self.fc3=nn.Linear(84,10)
        self.softmax1=nn.Softmax(dim=1)

    def forward(self, x):
        x=self.conv1(x)
        x=self.relu1(x)
        x=self.pool1(x)
        x=self.conv2(x)
        x=self.relu2(x)
        x=self.pool2(x)
        x=x.view(-1,16*5*5)
        x=self.fc1(x)
        x=self.relu3(x)
        x=self.fc2(x)
        x=self.relu4(x)
        x=self.fc3(x)
        x=self.softmax1(x)

        return x

# class MyDataSet(Dataset):
#      pass

def imshow(img):
    img = img / 2 + 0.5     # unnormalize
    npimg = img.numpy()
    plt.imshow(np.transpose(npimg, (1, 2, 0)))
    plt.show()

def visualize (loader, classes):

    for idx, data in enumerate(loader):
    # print images
        if idx > 0:
            return
        images, labels = data[0].to(device), data[1].to(device)
        #imshow(torchvision.utils.make_grid(images))
        print('GroundTruth: ', ' '.join('%5s' % classes[labels[j]] for j in range(4)))

        outputs = net(images)
        _,predicted=torch.max(outputs,1)

        print('Predicted: ', ' '.join('%5s' % classes[predicted[j]]
                                  for j in range(4)))




if __name__=='__main__':

    classes = ('plane', 'car', 'bird', 'cat',
               'deer', 'dog', 'frog', 'horse', 'ship', 'truck')

    transform = transforms.Compose(
        [transforms.ToTensor(),
         transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])

    trainset = torchvision.datasets.CIFAR10(root='./data', train=True,
                                            download=True, transform=transform)
    train_loader=DataLoader(dataset=trainset,batch_size=4,shuffle=True)

    testset = torchvision.datasets.CIFAR10(root='./data', train=False,
                                           download=True, transform=transform)
    test_loader=DataLoader(dataset=testset, batch_size=4,shuffle=False)

    device=torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    net=Net().to(device)

    criterion=nn.CrossEntropyLoss()
    optimizer=optim.SGD(params=net.parameters(),lr=0.001,momentum=0.9)

    # the following is to train the model
    for epoch in range (10):
        running_loss=0
        for idx, data in enumerate(train_loader):
            inputs, labels=data[0].to(device), data[1].to(device)
            optimizer.zero_grad()
            outputs=net(inputs)
            loss=criterion(outputs,labels)
            loss.backward()
            #optimizer.step is performs a parameter update based on the current gradient (stored in .grad attribute of a parameter
            optimizer.step()
            running_loss+=loss.item()

            if (idx+1)%2000==0:
                print ("epoch:",epoch, "idx:", idx+1, "ave loss: ",running_loss/2000)
                running_loss=0

    print ("training finished")

    visualize(test_loader, classes)

    #the following is to test the model
    total=0
    correct=0
    with torch.no_grad():
        for data in iter(test_loader):
            testimages,labels=data[0].to(device), data[1].to(device)
            outputs=net(testimages)
            loss=criterion(outputs,labels)
            _,predicted=torch.max(outputs,1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()

    accuracy=correct/total
    print ("accuracy on the test dataset: ",accuracy)