import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
import numpy as np
import math

# 0) Setting the data
traindata = '/Users/talgeller/Desktop/NumberDetector/train.csv'
testdata = '/Users/talgeller/Desktop/NumberDetector/test.csv'


class NumbersDataset(Dataset):

    def __init__(self, loadtxt):
        # data loading
        xy = np.loadtxt(loadtxt, delimiter=",", dtype=np.float32, skiprows=1)
        self.x = torch.from_numpy(xy[:, 1:])
        self.y = torch.from_numpy(xy[:, [0]])
        self.n_samples = xy.shape[0]

    def __getitem__(self, index):
        return self.x[index], self.y[index]

    def __len__(self):
        return self.n_samples


batch_size = 100
data_set = NumbersDataset(traindata)
dataloader = DataLoader(dataset=data_set, batch_size=batch_size, shuffle=True)


# 2) Implementing the module

class Numbersnn(nn.Module):
    def __init__(self, input_size, classes, hidden1, hidden2):
        super(Numbersnn, self).__init__()
        self.linear1 = nn.Linear(input_size, hidden1)
        self.relu = nn.ReLU()
        self.linear2 = nn.Linear(hidden1, hidden2)
        self.linear3 = nn.Linear(hidden2, classes)

    def forward(self, x):
        out = self.linear1(x)
        out = self.relu(out)
        out = self.linear2(out)
        out = self.relu(out)
        out = self.linear3(out)
        return out


hidden1 = 100
hidden2 = 32
input_size = 784
classes = 10
model = Numbersnn(input_size, classes, hidden1, hidden2)
# 3) Loss and optimizition

learning_rate = 0.00001
loss = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model.parameters(), learning_rate)

# 4) Training loop
epochs_loops = 1000
n_loops = math.floor(data_set.__len__() / batch_size) - 1
for epoch in range(epochs_loops):
    data_iter = iter(dataloader)
    data = next(data_iter)
    for i in range(n_loops):
        # forward
        images, lables = data
        lables = lables.view(100)
        lables = lables.long()
        ypred = model(images)
        l = loss(ypred, lables)

        # backward
        optimizer.zero_grad()
        l.backward()
        optimizer.step()

        data = next(data_iter)

# 5) Test data predictions

data_set_test = np.loadtxt(testdata, delimiter=",", dtype=np.float32, skiprows=1)
data_set_test = torch.from_numpy(data_set_test)
print(data_set_test.shape)
for i in range(data_set_test.__len__()):
    ypred = model(data_set_test[i])
    print(torch.argmax(ypred).item())
