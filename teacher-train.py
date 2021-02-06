#Copyright (C) 2019. Huawei Technologies Co., Ltd. All rights reserved.

#This program is free software; you can redistribute it and/or modify it under the terms of the BSD 3-Clause License.

#This program is distributed in the hope that it will be useful, but WITHOUT ANY WARRANTY; without even the implied warranty of MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the BSD 3-Clause License for more details.
import os
from torch.autograd import Variable
import argparse
from utils import *

set_seed(seed)

parser = argparse.ArgumentParser(description='train-teacher-network')

# Basic model parameters.
parser.add_argument('--dataset', type=str, default='MNIST', choices=['MNIST','SVHN','cifar10'])
parser.add_argument('--data', type=str, default='./Data/')
parser.add_argument('--output_dir', type=str, default='./Model/')
parser.add_argument('--model', type=str, default='LeNet5')
parser.add_argument('--teacher_name', type=str, default='teacher')
parser.add_argument('--output_file', type=str, default=os.path.join(os.path.dirname(__file__), 'results/') + 'results_teacher.csv')


args, unknown = parser.parse_known_args()
os.makedirs(args.output_dir, exist_ok=True)

if args.dataset == 'MNIST':
    data_train = MNIST(args.data,
                       transform=transforms.Compose([
                           transforms.Resize((32, 32)),
                           transforms.ToTensor(),
                           transforms.Normalize((0.1307,), (0.3081,))
                           ]),
                       download=True)  # True for the first time
    data_test = MNIST(args.data,
                      train=False,
                      transform=transforms.Compose([
                          transforms.Resize((32, 32)),
                          transforms.ToTensor(),
                          transforms.Normalize((0.1307,), (0.3081,))
                          ]))

    data_train_loader = DataLoader(data_train, batch_size=int(256), shuffle=True, num_workers=0)
    data_test_loader = DataLoader(data_test, batch_size=1024, num_workers=0)
if args.dataset == 'SVHN':
    mean = (0.4377, 0.4438, 0.4728)
    std = (0.1980, 0.2010, 0.1970)
    transform_train = transforms.Compose([
        transforms.Resize((32, 32)),
        transforms.ToTensor(),
        transforms.Normalize(mean, std)])
    transform_test = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(mean, std)])

    data_train = SVHN(args.data, split='train', download=True, transform=transform_train)
    data_test = SVHN(args.data, split='test', download=True, transform=transform_test)

    data_train_loader = DataLoader(data_train, batch_size=128, shuffle=True, num_workers=0)  # num_workers=8
    data_test_loader = DataLoader(data_test, batch_size=128, num_workers=0)  # num_workers=8
if args.dataset == 'cifar10':

    transform_train = transforms.Compose([
        transforms.RandomCrop(32, padding=4),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
    ])

    transform_test = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
    ])

    data_train = CIFAR10(args.data,
                       transform=transform_train,
                       download=True)  # True for the first time
    data_test = CIFAR10(args.data,
                      train=False,
                      transform=transform_test)

    data_train_loader = DataLoader(data_train, batch_size=128, shuffle=True, num_workers=0) # num_workers=8
    data_test_loader = DataLoader(data_test, batch_size=100, num_workers=0) # num_workers=8

if args.model == 'LeNet5':
    net = LeNet5().to(device)
    criterion = torch.nn.CrossEntropyLoss().to(device)
    optimizer = torch.optim.SGD(net.parameters(), lr=0.001)
elif args.model == 'WResNet40-2':
    net = WideResNet(depth=40, num_classes=10, widen_factor=2, dropRate=0.0).to(device)
    criterion = torch.nn.CrossEntropyLoss().to(device)
    optimizer = torch.optim.Adam(net.parameters(), lr=0.001)
elif args.model == 'ResNet34':
    net = resnet.ResNet34().to(device)
    criterion = torch.nn.CrossEntropyLoss().to(device)
    optimizer = torch.optim.SGD(net.parameters(), lr=0.001, momentum=0.9, weight_decay=5e-4)  # lr=0.1
else:
    raise Exception("Unknown teacher model name")


def adjust_learning_rate(optimizer, epoch):
    if epoch < 80:
        lr = 0.01
    elif epoch < 120:
        lr = 0.001
    else:
        lr = 0.0005
    for param_group in optimizer.param_groups:
        param_group['lr'] = lr


def model_train(epoch):
    if args.dataset == 'SVHN' or args.dataset == 'cifar10':
        adjust_learning_rate(optimizer, epoch)
    net.train()
    for i, (images, labels) in enumerate(data_train_loader):
        images, labels = Variable(images).to(device), Variable(labels).to(device)
        optimizer.zero_grad()
        output = net(images)
        loss = criterion(output, labels)

        if i == 1:
            print('Train - Epoch %d, TrainLoss: %f' % (epoch, loss.data.item()/data_train_loader.batch_size))
            with open(args.output_file, 'a') as f:
                f.write(str(epoch) + ',' + str(loss.data.item() / data_train_loader.batch_size) + ',')
                f.close()

        loss.backward()
        optimizer.step()
 
 
def model_test():
    net.eval()
    total_correct = 0
    avg_loss = 0.0
    with torch.no_grad():
        for i, (images, labels) in enumerate(data_test_loader):
            images, labels = Variable(images).to(device), Variable(labels).to(device)
            output = net(images)
            avg_loss += criterion(output, labels).sum()
            pred = output.data.max(1)[1]
            total_correct += pred.eq(labels.data.view_as(pred)).sum()

    avg_loss /= len(data_test)
    acc = float(total_correct) / len(data_test)
    print('TestLoss: %f, Accuracy: %f' % (avg_loss.data.item(), acc))
    with open(args.output_file, 'a') as f:
        f.write(str(avg_loss.data.item()) + ',' + str(acc) + '\n')
        f.close()
 
 
def train_and_test(epoch):
    model_train(epoch)
    model_test()

def main():
    if args.dataset == 'MNIST':
        epoch = 200
    else:
        epoch = 500
    with open(args.output_file, 'a') as f:
        f.write(
            'Epoch, TrainLoss, TestLoss, TestAccuracy \n')
        f.close()
    for e in range(1, epoch):
        train_and_test(e)
    torch.save(net, args.output_dir + args.teacher_name)
 
 
if __name__ == '__main__':
         main()

