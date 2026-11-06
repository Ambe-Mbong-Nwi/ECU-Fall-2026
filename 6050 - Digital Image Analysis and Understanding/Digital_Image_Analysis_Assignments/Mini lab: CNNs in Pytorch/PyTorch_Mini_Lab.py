import torch
import torch.nn as nn
import torchvision
import torchvision.transforms as transforms

from torch.utils.data import Dataset, DataLoader

transform = transforms.Compose([transforms.ToTensor(),
                                transforms.RandomCrop((256,256),pad_if_needed=True),
                                transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])

train_dataset = torchvision.datasets.OxfordIIITPet("pets_data/","trainval",transform=transform,download=True)
test_dataset = torchvision.datasets.OxfordIIITPet("pets_data/","test",transform=transform,download=True)

train_loader = DataLoader(train_dataset,batch_size=8)
test_loader = DataLoader(test_dataset,batch_size=8)

##### This network design is too shallow for the task, your job is to make it deeper! #####

class MyNetwork(nn.Module):
    def __init__(self):
        super(MyNetwork, self).__init__()
        self.ReLU = nn.ReLU()
        self.conv1 = nn.Conv2d(3, 64, 5, padding=2)
        self.conv2 = nn.Conv2d(64, 128, 3, padding=1)
        self.pool1 = nn.MaxPool2d(2, 2)
        self.conv3 = nn.Conv2d(128, 256, 3, padding=1)
        self.conv4 = nn.Conv2d(256, 256, 3, padding=1)
        self.pool2 = nn.MaxPool2d(2, 2)
        self.conv5 = nn.Conv2d(256, 512, 3, padding=1)
        self.pool3 = nn.MaxPool2d(2, 2)
        self.conv6 = nn.Conv2d(512, 512, 3, padding=1)
        self.fc1 = nn.Linear(512*32*32, 37)

    def forward(self, x):
        x = self.ReLU(self.conv1(x))
        x = self.ReLU(self.conv2(x))
        x = self.pool1(x)
        x = self.ReLU(self.conv3(x))
        x = self.ReLU(self.conv4(x))
        x = self.pool2(x)
        x = self.ReLU(self.conv5(x))
        x = self.pool3(x)
        x = self.ReLU(self.conv6(x))
        x = self.fc1(torch.flatten(x, start_dim=1))
        return x

#######################


model = MyNetwork()

##### Training Loop #####

learning_rate = 1e-4
optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
loss_func = torch.nn.CrossEntropyLoss()

num_epochs = 5

for epoch in range(num_epochs):
    print("Epoch:", epoch)
    for im, label, in train_loader:

        optimizer.zero_grad()

        output = model(im)
        loss = loss_func(output,label)
        print(loss)

        loss.backward()
        optimizer.step()


##### Testing Loop #####

correct = 0; total=0

model.eval()
with torch.no_grad():
    for im,label in test_loader:
        
        output = model(im)
        guess = torch.argmax(output,dim=1)
        
        correct += torch.sum(guess == label).item()
        total += len(guess)

print("Testing Accuracy is", correct/total * 100, "%")