import os
import sys
import json
import time
import torch
import torch.optim as optim
from torchvision import transforms, datasets, models
import torch.nn as nn
from utils import train_and_val,plot_acc,plot_loss
import numpy as np



if __name__ == '__main__':

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    print(device)

    if not os.path.exists('./results'):
        os.makedirs('./results')
    image_path = './datasets/'
    batch_size =32

    data_transform = {
        "train": transforms.Compose([transforms.Resize(256),
                                     transforms.RandomRotation(45),
                                     transforms.RandomHorizontalFlip(),
                                     transforms.ToTensor(),
                                     transforms.Normalize([0.624, 0.483, 0.415], [0.232, 0.200, 0.187])]),
        "val": transforms.Compose([transforms.Resize(256),
                                    transforms.ToTensor(),
                                    transforms.Normalize([0.624, 0.483, 0.415], [0.232, 0.200, 0.187])])}

    train_dataset = datasets.ImageFolder(root=image_path +"train", transform=data_transform["train"])  # 训练集数据
    train_loader = torch.utils.data.DataLoader(dataset=train_dataset, batch_size=batch_size, shuffle=True,
                                               num_workers=0)  # 加载数据
    len_train = len(train_dataset)
    val_dataset = datasets.ImageFolder(root=image_path +"val", transform=data_transform["val"])  # 测试集数据
    val_loader = torch.utils.data.DataLoader(dataset=val_dataset, batch_size=batch_size, shuffle=False,
                                             num_workers=0)  # 加载数据
    len_val = len(val_dataset)

    net = models.resnet50(pretrained=True)
    inchannel = net.fc.in_features
    # net.fc = nn.Linear(inchannel, 5)
    # net = models.vgg16(pretrained=True)
    # net = models.alexnet(pretrained=True)
    # nn.Linear(in_features=512, out_features=5)

    net.fc = nn.Sequential(
        nn.ReLU(),
        nn.Dropout(0.5),
        nn.LogSoftmax(dim=1)
    )

    loss_function = nn.CrossEntropyLoss()  # loss function
    optimizer = optim.Adam(net.parameters(), lr=0.00005, betas=(0.9, 0.999), eps=1e-08,
                           weight_decay=0,
                           amsgrad=False)

    net.to(device)
    epoch = 60

    history = train_and_val(epoch, net, train_loader, len_train,val_loader, len_val,loss_function, optimizer,device)

    plot_loss(np.arange(0,epoch), history)
    plot_acc(np.arange(0,epoch), history)



