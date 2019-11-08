import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim



#nn.Module is the Base class for all neural network modules.

class Net (nn.Module):

    def __init__(self):
        super(Net, self).__init__()

        self.conv1=nn.Conv2d(1,6,3)
        self.relu1=nn.ReLU(inplace=True)
        self.max_pool1=nn.MaxPool2d(kernel_size=2)

        self.conv2=nn.Conv2d(6,16,3)
        self.relu2=nn.ReLU(inplace=True)
        self.max_pool2=nn.MaxPool2d(kernel_size=2)

        self.fc1=nn.Linear(16*6*6,120)
        self.relu3=nn.ReLU(inplace=True)
        self.fc2=nn.Linear(120,84)
        self.relu4=nn.ReLU(inplace=True)
        self.fc3=nn.Linear(84,10)

    def forward(self, x):
        x=self.conv1(x)
        x=self.relu1(x)
        x=self.max_pool1(x)

        x=self.conv2(x)
        x=self.relu2(x)
        x=self.max_pool2(x)

        x = x.view(-1, self.num_flat_features(x))

        x=self.fc1(x)
        x=self.relu3(x)
        x=self.fc2(x)
        x=self.relu4(x)
        x=self.fc3(x)


        return x

    def num_flat_features(self, x):
        size = x.size()[1:]  # all dimensions except the batch dimension
        num_features = 1
        for s in size:
            num_features *= s
        return num_features

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
print(device)

 #batch size, channel, width height
input=torch.randn(1,1,32,32).to(device)

# visualize_dataset(train_dataset)
net = Net().to(device)

target = torch.rand(1,10).to(device)
print (target)

criterion=nn.MSELoss()
optimizer=optim.SGD(params=net.parameters(),lr=0.01)

for i in range (100):
    optimizer.zero_grad()
    output=net(input)
    loss=criterion(output, target)
    print (i," loss:", loss)
    loss.backward()
    optimizer.step()

