import torchvision.transforms as transforms
from torch.utils.data import DataLoader
from torchvision.datasets.mnist import MNIST
from lenet import *
from torchvision.datasets import SVHN
from torchvision.datasets import CIFAR10
import resnet
from wresnet import *
import numpy
import random
import matplotlib.pyplot as plt

use_gpu = True
use_gpu = use_gpu and torch.cuda.is_available()
device = torch.device('cuda') if use_gpu else torch.device('cpu')

seed = 1

def set_seed(seed):
    random.seed(seed)
    numpy.random.seed(seed)  # numpy
    torch.manual_seed(seed)  # cpu
    torch.cuda.manual_seed(seed)  # gpu
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True  # cudnn

def kdloss(y, teacher_scores):
    p = F.log_softmax(y, dim=1)
    q = F.softmax(teacher_scores, dim=1)
    l_kl = F.kl_div(p, q, reduction='sum') / y.shape[0]
    return l_kl


def load_student(stu_model, lr_S, optm_S= 'Adam', s_momentum=0, s_weight_decay=0, teacher_dataset= None):
    if stu_model == 'LeNet5':
        student_net = LeNet5().to(device)
        student_net = nn.DataParallel(student_net).to(device)
        if optm_S == 'SGD':
            optimizer_S = torch.optim.SGD(student_net.parameters(), lr=lr_S, momentum=s_momentum, weight_decay=s_weight_decay)
        elif optm_S == 'Adam':
            optimizer_S = torch.optim.Adam(student_net.parameters(), lr=lr_S, weight_decay=s_weight_decay)  # better
        else:
            raise Exception("Unknown student optimizer")

    elif stu_model == 'LeNet5Half':
        student_net = LeNet5Half().to(device)
        student_net = nn.DataParallel(student_net).to(device)
        if optm_S == 'SGD':
            optimizer_S = torch.optim.SGD(student_net.parameters(), lr=lr_S, momentum=s_momentum, weight_decay=s_weight_decay)
        elif optm_S == 'Adam':
            optimizer_S = torch.optim.Adam(student_net.parameters(), lr=lr_S, weight_decay=s_weight_decay)  #
        else:
            raise Exception("Unknown student optimizer")

    elif stu_model == 'WResNet16-1':
        student_net = WideResNet(depth=16, num_classes=10, widen_factor=1, dropRate=0.0).to(device)
        student_net = nn.DataParallel(student_net).to(device)
        if optm_S == 'SGD':
            optimizer_S = torch.optim.SGD(student_net.parameters(), lr=lr_S, momentum=s_momentum, weight_decay=s_weight_decay)
        elif optm_S == 'Adam':
            optimizer_S = torch.optim.Adam(student_net.parameters(), lr=lr_S, weight_decay=s_weight_decay)  #
        else:
            raise Exception("Unknown student optimizer")
        
    elif stu_model == 'ResNet18':
        if teacher_dataset == 'cifar10':
            student_net = resnet.ResNet18().to(device)
            student_net = nn.DataParallel(student_net).to(device)
        if teacher_dataset == 'cifar100':
            student_net = resnet.ResNet18(num_classes=100).to(device)
            student_net = nn.DataParallel(student_net).to(device)
        if optm_S == 'SGD':
            optimizer_S = torch.optim.SGD(student_net.parameters(), lr=lr_S, momentum=s_momentum, weight_decay=s_weight_decay) # momentum=0.9, weight_decay=5e-4
        elif optm_S == 'Adam':
            optimizer_S = torch.optim.Adam(student_net.parameters(), lr=lr_S, weight_decay=s_weight_decay)
        else:
            raise Exception("Unknown student optimizer")

    else:
        raise Exception("Unknown student model name")
    return student_net, optimizer_S


def load_data_test_loader(data, dataset_teacher, batch_size=512):
    if dataset_teacher == 'MNIST':
        # Configure data loader
        data_test = MNIST(data,
                          train=False,
                          transform=transforms.Compose([
                              transforms.Resize((32, 32)),
                              transforms.ToTensor(),
                              transforms.Normalize((0.1307,), (0.3081,))
                          ]))
        # data_test_loader = DataLoader(data_test, batch_size=64, num_workers=1, shuffle=False)
        data_test_loader = DataLoader(data_test, batch_size=batch_size, shuffle=False)
    elif dataset_teacher == 'SVHN':
        mean = (0.4377, 0.4438, 0.4728)
        std = (0.1980, 0.2010, 0.1970)
        transform_test = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize(mean, std)])
        data_test = SVHN(data, split='test', download=False, transform=transform_test)
        data_test_loader = DataLoader(data_test, batch_size=batch_size, shuffle=False)
    elif dataset_teacher == 'cifar10':
        transform_test = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
        ])
        data_test = CIFAR10(data,
                            train=False,
                            transform=transform_test)
        data_test_loader = DataLoader(data_test, batch_size=batch_size, num_workers=0)
    else:
        raise Exception("Unknown dataset_teacher name")
    return data_test_loader, data_test


def load_data_train_loader(data, dataset_teacher, batch_size=512, shuffle=False):
    if dataset_teacher == 'MNIST':
        # Configure data loader
        data_train = MNIST(data,
                           transform=transforms.Compose([
                               transforms.Resize((32, 32)),
                               transforms.ToTensor(),
                               transforms.Normalize((0.1307,), (0.3081,))
                           ]),
                           download=False)  # True for the first time
        data_train_loader = DataLoader(data_train, batch_size=batch_size, shuffle=shuffle, num_workers=0)  # num_workers=8
    elif dataset_teacher == 'SVHN':
        mean = (0.4377, 0.4438, 0.4728)
        std = (0.1980, 0.2010, 0.1970)
        transform_train = transforms.Compose([
            transforms.Resize((32, 32)),
            transforms.ToTensor(),
            transforms.Normalize(mean, std)])
        data_train = SVHN(data, split='train', download=False, transform=transform_train)
        data_train_loader = DataLoader(data_train, batch_size=batch_size, shuffle=shuffle, num_workers=0)  # num_workers=8
    elif dataset_teacher == 'cifar10':
        transform_train = transforms.Compose([
            transforms.RandomCrop(32, padding=4),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
        ])
        data_train = CIFAR10(data,
                             transform=transform_train,
                             download=False)  # True for the first time
        data_train_loader = DataLoader(data_train, batch_size=batch_size, shuffle=shuffle, num_workers=0)  # num_workers=8
    else:
        raise Exception("Unknown dataset_teacher name")
    return data_train_loader, data_train

def model_test(net, data_test, data_test_loader, criterion):
    avg_loss = 0.0
    total_correct = 0
    avg_confidence = 0.0
    with torch.no_grad():
        for i, (images, labels) in enumerate(data_test_loader):
            images = images.to(device)
            labels = labels.to(device)
            net.eval()
            output = net(images)
            avg_loss += criterion(output, labels).sum()
            pred = output.data.max(1)[1]
            total_correct += pred.eq(labels.data.view_as(pred)).sum()
            avg_confidence += torch.nn.functional.softmax(output, dim=1).max(1)[0].sum()

    avg_loss /= len(data_test)
    avg_confidence /= len(data_test)
    # print('Test Avg. Loss: %f, Accuracy: %f' % (avg_loss.data.item(), float(total_correct) / len(data_test)))
    accr = round(float(total_correct) / len(data_test), 4)
    return avg_loss.data.item(), accr, avg_confidence.item()
    


def adjust_learning_rate(optimizer, epoch, learing_rate):
    if epoch < 800:
        lr = learing_rate
    elif epoch < 1600:
        lr = 0.1 * learing_rate
    else:
        lr = 0.01 * learing_rate
    for param_group in optimizer.param_groups:
        param_group['lr'] = lr



def im2tensor(image, imtype=numpy.uint8, cent=1., factor=255./2.):
    return torch.Tensor((image / factor - cent)
                        [:, :, :, numpy.newaxis].transpose((3, 2, 0, 1)))


def load_image(path):
    if(path[-3:] == 'dng'):
        import rawpy
        with rawpy.imread(path) as raw:
            img = raw.postprocess()
    elif(path[-3:]=='bmp' or path[-3:]=='jpg' or path[-3:]=='png'):
        import cv2
        return cv2.imread(path)[:,:,::-1]
    else:
        img = (255*plt.imread(path)[:,:,:3]).astype('uint8')
    return img
