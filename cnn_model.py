import torch.nn as nn
import torch.nn.functional as f
import torch


class Net(nn.Module):

    def __init__(self):

        super(Net, self).__init__()
        self.conv1 = nn.Conv2d(3, 32, 3)
        self.conv2 = nn.Conv2d(32, 64, 3)
        self.fc1 = nn.Linear(33856, 1024)
        self.fc2 = nn.Linear(1024, 512)
        self.fc3 = nn.Linear(512, 28)

    def forward(self, x):
        x = f.dropout2d(f.max_pool2d(f.relu(self.conv1(x)), 2), 0.5)
        x = f.dropout2d(f.max_pool2d(f.relu(self.conv2(x)), 2), 0.5)
        x = x.view(-1, self.num_flat_features(x))
        x = f.dropout(f.relu(self.fc1(x)), 0.5)
        x = f.relu(self.fc2(x))
        x = self.fc3(x)
        return x

    @staticmethod
    def num_flat_features(x):
        size = x.size()[1:]
        num_features = 1
        for s in size:
            num_features *= s
        return num_features


def train(model, device, train_loader, optimizer, epoch):
    model.train()
    acc = 0.0
    for batch_idx, sample in enumerate(train_loader):
        data, target = sample['image'], sample['keypoints']
        data, target = data.float().to(device), target.float().to(device)
        # print(data.size(), target.size())
        optimizer.zero_grad()
        output = model(data)
        # print(output.size())
        loss = nn.MSELoss()
        out = loss(output, target)
        out.backward()
        optimizer.step()
        acc += batch_acc(output, target)
        if batch_idx % 5 == 0:
            print('Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}\tAccuracy: {:.2f}'.format(
                epoch, batch_idx * len(data), len(train_loader.dataset),
                100. * batch_idx / len(train_loader), out.item(), acc/5))
            acc = 0.0


def test(model, device, test_loader):
    model.eval()
    test_loss = 0
    correct = 0
    acc = 0.0
    with torch.no_grad():
        for i, sample in enumerate(test_loader):
            data, target = sample['image'], sample['keypoints']
            data, target = data.float().to(device), target.float().to(device)
            output = model(data)
            loss = nn.MSELoss()
            test_loss += loss(output, target).item()
            acc += batch_acc(output, target)
            correct += correct_pred(output, target)
    test_loss /= 600
    acc = (acc*100)/(i+1)
    print('\nTest set: Average loss: {:.4f}, Accuracy:({}%)\n'.format(
        test_loss, acc))

def batch_acc(pred, gt):
    if pred.size()[0] != gt.size()[0]:
        pred = pred[:gt.size()[0]]

    pred, gt = pred.view(-1, 14, 2), gt.view(-1, 14, 2)
    euc_distance = torch.sqrt(torch.sub(pred, gt).pow(2).sum(2))
    threshold = 5
    acc = torch.mean((euc_distance < threshold).float(), 1)
    batch_avg_acc = torch.mean(acc)
    return batch_avg_acc.item()


def correct_pred(pred, gt):
    if pred.size()[0] != gt.size()[0]:
        pred = pred[:gt.size()[0]]

    pred, gt = pred.view(-1, 14, 2), gt.view(-1, 14, 2)
    euc_distance = torch.sqrt(torch.sub(pred, gt).pow(2).sum(2))
    min_correct_pred = 12
    threshold = 5
    pred = torch.sum((euc_distance < threshold).float(), 1)
    total_correct = torch.sum((pred > min_correct_pred).float())
    return total_correct.item()
