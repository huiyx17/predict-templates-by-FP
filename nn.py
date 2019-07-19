import pickle
import torch
import torch.nn as nn

import torch.nn.functional as F
import torch.optim as optim
from torchvision import datasets, transforms
from torch.autograd import Variable
import numpy as np
from torch.utils.data import Dataset, DataLoader
with open('train_test_dataset.pickle','rb') as f:
    FP_train,FP_test, labels_train, labels_test = pickle.load(f)
devices = torch.device('cuda')
#cuda_gpu = torch.cuda.is_available()
num_classes = max(labels_train)

SEED = 0
torch.manual_seed(SEED)
torch.cuda.manual_seed(SEED)

class FPDataset(Dataset):
    def __init__(self,FP,labels,transform=None):
        self.transform = transform
        self.FP = torch.LongTensor(FP).cuda(device = devices)
        self.labels = torch.LongTensor(labels).cuda(device = devices)
        #self.FP = torch.LongTensor(FP) 
        #self.labels = torch.LongTensor(labels)
    def __len__(self):
        return len(self.FP)
    def __getitem__(self,idx):
        f, l = self.FP[idx], self.labels[idx]
        return f, l-1

traindataset = FPDataset(FP_train, labels_train)
trainloader = DataLoader(traindataset, batch_size=128,shuffle=True)
testdataset = FPDataset(FP_test, labels_test)
testloader = DataLoader(testdataset, batch_size=128,shuffle=True)

class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        self.l1 = nn.Linear(1024, 300)
        self.l2 = nn.Linear(300, 300)
        self.l3 = nn.Linear(300, 300)
        self.l4 = nn.Linear(300, 300)
        self.l5 = nn.Linear(300,num_classes )

    def forward(self, x):
        # Flatten the data (n, 1, 28, 28) --> (n, 784)
        x = x.view(-1, 1024)
        x = F.relu(self.l1(x))
        x = F.relu(self.l2(x))
        x = F.relu(self.l3(x))
        x = F.relu(self.l4(x))
        return F.log_softmax(self.l5(x), dim=1)
        #return self.l5(x)
model = Net().cuda(device = devices)
#model = Net()
optimizer = optim.SGD(model.parameters(), lr=0.005)

def train(epoch):
    model.train()
    for batch_idx, (data, target) in enumerate(trainloader):

        data, target = Variable(data.float()), Variable(target)
        optimizer.zero_grad()
        output = model(data)
        # loss
        loss = F.nll_loss(output, target)
        loss.backward()
        # update
        optimizer.step()
        if batch_idx % 200 == 0:
            print('Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}'.format(
                epoch, batch_idx * len(data), len(trainloader.dataset),
                100. * batch_idx / len(trainloader), loss.data))

def test(epoch):
    model.eval()
    total = 0
    correct = 0
    for data, target in testloader:
        data, target = Variable(data.float()), Variable(target)
        output = model(data)
        _, predicted = torch.topk(output, 25, 1, True, True)
        for i in range(len(target)):
            if target[i] in predicted[i]:
                correct += 1
        total += target.size(0)   
    print('Accuracy of the network %d' % (100 * correct / total), correct, total)

for epoch in range(1,200):
    train(epoch)
    test(epoch)
torch.save(model.state_dict(), 'linear_model')