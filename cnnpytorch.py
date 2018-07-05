from torchvision import transforms, utils
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from cnn_model import Net, train, test
from dataload import *
from data_transform import *
from data_utils import *
from torch.utils.data.sampler import SubsetRandomSampler

face_dataset = FacialKeypointsDataset(filename='data/gt.csv', root_dir='./data/images/',
                                      transform=transforms.Compose([flip2d(), ToTensor()]))

validation_split = 0.1
shuffle_dataset = True
random_seed= 42

# Creating data indices for training and validation splits:
dataset_size = len(face_dataset)
indices = list(range(dataset_size))
split = int(np.floor(validation_split * dataset_size))
if shuffle_dataset:
    np.random.seed(random_seed)
    np.random.shuffle(indices)
train_indices, val_indices = indices[split:], indices[:split]

# Creating PT data samplers and loaders:
train_sampler = SubsetRandomSampler(train_indices)
valid_sampler = SubsetRandomSampler(val_indices)

#dataloader = DataLoader(face_dataset, batch_size=32, shuffle=True)
train_loader = torch.utils.data.DataLoader(face_dataset, batch_size=32, sampler=train_sampler)
validation_loader = torch.utils.data.DataLoader(face_dataset, batch_size=32, sampler=valid_sampler)

use_cuda = torch.cuda.is_available()
device = torch.device("cuda" if use_cuda else "cpu")
print(device)

net = Net().to(device)
print(net)


# params = list(net.parameters())
# print(len(params))
# for i in range(10):
#     print(params[i].size())

# optimizer = optim.SGD(net.parameters(), lr=0.001, momentum=0.9)
optimizer = optim.Adam(net.parameters())
epochs = 20

# net.load_state_dict(torch.load('model.net'))

for epoch in range(1, epochs + 1):
    train(net, device, train_loader, optimizer, epoch)
    test(net, device, validation_loader)

# torch.save(net.state_dict(), "model.net")

for i_batch, sample_batched in enumerate(validation_loader):
    net.eval()
    data, target = sample_batched['image'].float().to(device), sample_batched['keypoints'].float().to(device)
    out = net(data)
    if i_batch == 1:
        for i in range(5):
            plt.figure()
            plt.imshow(data[i].cpu().numpy().transpose(1, 2, 0))
            target_i = target[i].cpu().numpy().reshape(-1, 2)
            out_i = out[i].cpu().detach().numpy().reshape(-1, 2)
            plt.scatter(out_i[:, 0], out_i[:, 1], s=10, marker='.', c='r')
            # plt.scatter(target_i[:, 0], target_i[:, 1], s=10, marker='.', c='b')
            plt.axis('off')
            plt.show()
        break

print(target_i)

test(net, device, validation_loader)
