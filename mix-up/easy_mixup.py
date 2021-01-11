from __future__ import print_function

import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import torch.backends.cudnn as cudnn




import torchvision
import torchvision.transforms as transforms

import os
import argparse
import csv



from models import *
from utils import progress_bar, mixup_data, \
    mixup_criterion, mul_mixup_data, mul_mixup_criterion, \
    exhaustive_mix_data_pre, exhausitive_mix_data, hyper_mixup_criterion
from torch.autograd import Variable


import datetime


parser = argparse.ArgumentParser(description='PyTorch CIFAR10 Training')
parser.add_argument('--resume', '-r', default=False, action='store_true', help='resume from checkpoint')
parser.add_argument('--sess', default='x', type=str, help='session id')
parser.add_argument('--seed', default=0, type=int, help='rng seed')
parser.add_argument('--alpha', default=1., type=float, help='interpolation strength (uniform=1., ERM=0.)')
parser.add_argument('--decay', default=1e-4, type=float, help='weight decay (default=1e-4)')
args = parser.parse_args()

torch.manual_seed(args.seed)


"""
use single page mode
"""
use_multi_page = False
run_online = True

# run_mode = "none"
run_mode = "hyper_search"

data_path = "./data"
check_point_path = "/home/ubuntu/MyFiles/checkpoint"


if run_online:
    data_path = "/home/ubuntu/MyFiles/data"

# check_point_path = './checkpoint'
# data_path = './data'


use_cuda = torch.cuda.is_available()
best_acc = 0  # best test accuracy
start_epoch = 0  # start from epoch 0 or last checkpoint epoch
batch_size = 128
base_learning_rate = 0.1
if use_cuda:
    # data parallel
    n_gpu = torch.cuda.device_count()
    batch_size *= n_gpu
    base_learning_rate *= n_gpu


# Data
print('==> Preparing data..')
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

trainset = torchvision.datasets.CIFAR10(root=data_path, train=True, download=True, transform=transform_train)
trainloader = torch.utils.data.DataLoader(trainset, batch_size=batch_size, shuffle=True, num_workers=2)

testset = torchvision.datasets.CIFAR10(root=data_path, train=False, download=True, transform=transform_test)
testloader = torch.utils.data.DataLoader(testset, batch_size=100, shuffle=False, num_workers=0)


classes = ('plane', 'car', 'bird', 'cat', 'deer', 'dog', 'frog', 'horse', 'ship', 'truck')

#Model
if args.resume:
    # Load checkpoint.
    print('==> Resuming from checkpoint..')
    assert os.path.isdir(check_point_path), 'Error: no checkpoint directory found!'
    checkpoint = torch.load(check_point_path + '/ckpt.t7.' + args.sess + '_' + str(args.seed))
    net = checkpoint['net']
    best_acc = checkpoint['acc']
    start_epoch = checkpoint['epoch'] + 1
    torch.set_rng_state(checkpoint['rng_state'])
else:
    print('==> Building model..')
    # net = VGG('VGG19')
    # net = PreActResNet18()
    net = ResNet18()


result_folder = './results/'
if not os.path.exists(result_folder):
    os.makedirs(result_folder)

logname = result_folder + net.__class__.__name__ + "_" + args.sess + str(args.seed) + '.csv'


data_folder = './log/'
if not os.path.exists(data_folder):
    os.makedirs(data_folder)
data_log_name = data_folder + net.__class__.__name__ + "_data_" + args.sess + str(args.seed) + '.csv'


if use_cuda:
    net.cuda()
    net = torch.nn.DataParallel(net)
    print('Using', torch.cuda.device_count())
    cudnn.benchmark = True
    print("Using CUDA..")

criterion_test = torch.nn.CrossEntropyLoss()

if use_multi_page:
    criterion = nn.CrossEntropyLoss(reduction="none")
    # criterion = FocalLoss(num_class=10)
    # criterion = CustomLoss()
else:
    criterion = nn.CrossEntropyLoss(reduction="none")



# Tracking
def train(epoch):
    print('\n Epoch: %d' % epoch)
    net.train()
    train_loss = 0
    correct = 0
    total = 0
    for batch_idx, (inputs, targets) in enumerate(trainloader):
        if use_cuda:
            inputs, targets = inputs.cuda(), targets.cuda()
        # generate mixed inputs, two one-hot label vectors and mixing coefficient

        if use_multi_page:

            inputs, targets_a, targets_b, lam = mul_mixup_data(inputs, targets, args.alpha, use_cuda)
            optimizer.zero_grad()
            inputs, targets_a, targets_b = Variable(inputs), Variable(targets_a), Variable(targets_b)
            outputs = net(inputs)
            loss_func = mul_mixup_criterion(targets_a, targets_b, lam)
        else:
            inputs, targets_a, targets_b, lam = mixup_data(inputs, targets, args.alpha, use_cuda)
            optimizer.zero_grad()
            inputs, targets_a, targets_b = Variable(inputs), Variable(targets_a), Variable(targets_b)
            outputs = net(inputs)

            loss_func = mixup_criterion(targets_a, targets_b, lam)

        loss = loss_func(criterion, outputs)
        loss.backward()
        optimizer.step()

        train_loss += loss.data.item()
        _, predicted = torch.max(outputs.data, 1)
        total += targets.size(0)
        correct += lam * predicted.eq(targets_a.data).cpu().sum() + (1 - lam) * predicted.eq(targets_b.data).cpu().sum()
        correct = correct.item()
        progress_bar(batch_idx, len(trainloader), 'Loss: %.3f | Acc: %.3f%% (%d/%d)'
                     % (train_loss / (batch_idx + 1), 100. * correct / total, correct, total))
    return (train_loss / batch_idx, 100. * correct / total)


"""
exhaustive searching
"""

def train_with_exhaustive_testing(epoch):
    """
    to find best pair for a single picture
    :param epoch: epoch to train
    :return: accuracy
    """
    print('\n Epoch: %d' % epoch)
    train_loss = 0
    correct = 0
    total = 0
    # et_criterion = CustomLoss()
    et_criterion = nn.CrossEntropyLoss(reduction='none')
    for batch_idx, (inputs, targets) in enumerate(trainloader):
        if use_cuda:
            inputs, targets = inputs.cuda(), targets.cuda()

        net.eval()
        best_pair_list = None
        lam_list = None
        with torch.no_grad():
            # starttime = datetime.datetime.now()

            for index in range(inputs.size()[0]):

                inputs_next, targets_a, targets_b, lam = exhaustive_mix_data_pre(inputs, targets, index, use_cuda=use_cuda)
                inputs_next, targets_a, targets_b = Variable(inputs_next), Variable(targets_a), Variable(targets_b)
                outputs = net(inputs_next)
                loss_func = mixup_criterion(targets_a, targets_b, lam)
                best_pair = loss_func(et_criterion, outputs)
                best_pair = best_pair.argmax()
                if type(best_pair_list) == type(None):
                    best_pair_list = best_pair.unsqueeze(0)
                    lam_list = torch.tensor([lam], dtype=torch.float)
                else:
                    best_pair_list = torch.cat((best_pair_list, best_pair.unsqueeze(0)),0)
                    lam = torch.tensor([lam], dtype=torch.float)
                    lam_list = torch.cat((lam_list, lam),0)

            # find_pair_time = datetime.datetime.now()
            # print(str((find_pair_time - starttime).microseconds) + "======")

        net.train()
        with torch.enable_grad():
            optimizer.zero_grad()
            lam_list = lam_list.cuda()
            inputs, targets_a, targets_b, lam = exhausitive_mix_data(inputs, best_pair_list, targets, lam_list)
            loss_func = mixup_criterion(targets_a, targets_b, lam)
            outputs = net(inputs)
            loss = loss_func(criterion, outputs)
            loss.backward()
            optimizer.step()

        # train_time = datetime.datetime.now()

        # print(str((train_time - find_pair_time).microseconds) + "--------")

        train_loss += loss.data.item()
        _, predicted = torch.max(outputs.data, 1)
        total += targets.size(0)
        # correct += lam * predicted.eq(targets_a.data).cpu().sum() + (1 - lam) * predicted.eq(targets_b.data).cpu().sum()
        # correct = correct.item()
        correct = 0
        print(batch_idx, len(trainloader), 'Loss: %.3f | Acc: %.3f%% (%d/%d)'
                     % (train_loss / (batch_idx + 1), 100. * correct / total, correct, total))
    return (train_loss / batch_idx, 100. * correct / total)



"""
parameter training
"""

def hyper_train(epoch):
    print('\n Epoch: %d' % epoch)
    net.train()
    train_loss = 0
    correct = 0
    total = 0
    for batch_idx, (inputs, targets) in enumerate(trainloader):
        if use_cuda:
            inputs, targets = inputs.cuda(), targets.cuda()

        batch_size = inputs.size()[0]

        weight_matrix = get_adversary(net, inputs, targets)
        # print("\n")
        # print(weight_matrix.tolist()[0])
        # print("----")
        # print(targets[0])
        # print("====")

        """normalization"""
        # if torch.min(weight_matrix) < 0:
        #     weight_matrix = weight_matrix - torch.min(weight_matrix)
        # weight_matrix =zz weight_matrix/torch.max(weight_matrix)

        # relu = nn.ReLU()
        # weight_matrix = relu(weight_matrix)
        # weight_matrix = F.normalize(weight_matrix, p=1, dim=1)
        # print(weight_matrix)


        """use softmax"""
        # relu = nn.ReLU()
        # weight_matrix = relu(weight_matrix)
        # softmax = nn.Softmax()
        # weight_matrix = softmax(weight_matrix)


        """use top N item"""
        top_4 = weight_matrix.topk(4)[1]

        main_page = top_4[:, 0].reshape(batch_size, 1)
        main_page_values = torch.full([batch_size, batch_size], 0.5).cuda()

        sub_page = top_4[:, 1].reshape(batch_size, 1)
        sub_page_values = torch.full([batch_size, batch_size], 0.2).cuda()

        third_page = top_4[:, 2].reshape(batch_size, 1)
        third_page_values = torch.full([batch_size, batch_size], 0.2).cuda()

        fourth_page = top_4[:, 3].reshape(batch_size, 1)
        fourth_page_values = torch.full([batch_size, batch_size], 0.1).cuda()

        weight_matrix.scatter_(1, main_page, main_page_values)
        weight_matrix.scatter_(1, sub_page, sub_page_values)
        weight_matrix.scatter_(1, third_page, third_page_values)
        weight_matrix.scatter_(1, fourth_page, fourth_page_values)

        weight_matrix_zeros = torch.zeros_like(weight_matrix).cuda()
        weight_matrix = torch.where(weight_matrix < 0.1, weight_matrix_zeros, weight_matrix)

        """end"""

        # weight_matrix = torch.sigmoid(weight_matrix)

        # weight_matrix = torch.eye(batch_size, requires_grad=True).cuda()
        new_x = inputs.reshape(batch_size, -1)
        new_x = torch.mm(weight_matrix, new_x)
        new_x = new_x.reshape(batch_size, 3, 32, 32)

        optimizer.zero_grad()
        outputs = net(new_x)
        value = 0
        for index in range(0, 10):
            sub_matrix = weight_matrix[:,(targets==index).nonzero()].sum(1)
            item = torch.full([batch_size], index).cuda().long()
            loss = criterion(outputs, item)
            value += sub_matrix.reshape(batch_size)*loss

        loss = value.mean()
        loss.backward()
        optimizer.step()


        train_loss += loss.data.item()
        _, predicted = torch.max(outputs.data, 1)
        total += targets.size(0)
        correct += 0
        # correct = correct.item()
        progress_bar(batch_idx, len(trainloader), 'Loss: %.3f | Acc: %.3f%% (%d/%d)'
                     % (train_loss / (batch_idx + 1), 100. * correct / total, correct, total))

        if epoch % 100 == 0:

            with open(args.data_log_name, 'a') as logfile:
                logwriter = csv.writer(logfile, delimiter=',')
                logwriter.writerow(weight_matrix.tolist())
    return (train_loss / batch_idx, 100. * correct / total)






def test(epoch):
    global best_acc
    net.eval()
    test_loss = 0
    correct = 0
    total = 0
    for batch_idx, (inputs, targets) in enumerate(testloader):
        if use_cuda:
            inputs, targets = inputs.cuda(), targets.cuda()
        with torch.no_grad():
            outputs = net(inputs)
            loss = criterion_test(outputs, targets)

            test_loss += loss.item()
            _, predicted = torch.max(outputs.data, 1)
            total += targets.size(0)
            correct += predicted.eq(targets.data).cpu().sum()
            print(batch_idx, len(testloader), 'Loss: %.3f | Acc: %.3f%% (%d/%d)'
                % (test_loss/(batch_idx+1), 100.*correct/total, correct, total))
    correct = correct.item()
    # Save checkpoint.
    acc = 100.*correct/total
    # if acc > best_acc:
    #     best_acc = acc
    checkpoint(acc, epoch)
    return (test_loss/batch_idx, 100.*correct/total)



def checkpoint(acc, epoch):
    # Save checkpoint.
    print('Saving..')
    state = {
        'net': net,
        'acc': acc,
        'epoch': epoch,
        'rng_state': torch.get_rng_state()
    }
    if not os.path.isdir(check_point_path):
        os.mkdir(check_point_path)
    torch.save(state, check_point_path + '/ckpt.t7.' + args.sess + '_' + str(args.seed))

def adjust_learning_rate(optimizer, epoch):
    """decrease the learning rate at 100 and 150 epoch"""
    lr = base_learning_rate
    if epoch <= 9 and lr > 0.1:
        # warm-up training for large minibatch
        lr = 0.1 + (base_learning_rate - 0.1) * epoch / 10.
    if epoch >= 100:
        lr /= 10
    if epoch >= 150:
        lr /= 10
    for param_group in optimizer.param_groups:
        param_group['lr'] = lr


def get_adversary(model, x, y):
    # sudo_label = model(x).max(1)[1] # this should be change to model(x)[:class]
    true_label = y
    k = args.sub_k
    alpha = args.al
    # print(alpha)

    batch_size = x.size()[0]

    with torch.enable_grad():  # because usually people disables gradients in evaluations
        # This could be chazeros_likenged to randn, but lets see
        weight_matrix = torch.eye(batch_size, requires_grad=True).cuda()

        for _ in range(k):
            new_x = x.reshape(batch_size, -1)
            new_x = torch.mm(weight_matrix, new_x)
            new_x = new_x.reshape(batch_size, 3, 32, 32)
            loss = F.cross_entropy(model(new_x), true_label)
            grad = torch.autograd.grad(loss, weight_matrix)
            weight_matrix.data += alpha * grad[0]

            # weight_matrix.data -= alpha * grad[0].sign()
    return weight_matrix




"""
main function
"""
if not os.path.exists(logname):
    with open(logname, 'w') as logfile:
        logwriter = csv.writer(logfile, delimiter=',')
        logwriter.writerow(['epoch', 'train loss', 'train acc', 'test loss', 'test acc'])

hyper = [(1, 0.002), (1, 0.02)]

for item in hyper:
    args.sub_k = item[0]
    args.al = item[1]

    logname = result_folder + net.__class__.__name__ + "_k_" + \
              str(item[0]) + "_alpha_" + str(item[1]) + "_" + args.sess + str(args.seed) + '.csv'

    args.data_log_name = data_folder + net.__class__.__name__ + "_k_" + \
              str(item[0]) + "_alpha_" + str(item[1]) + "_" + args.sess + str(args.seed) + '.csv'


    net = ResNet18()

    # net.cuda()
    net = torch.nn.DataParallel(net)
    cudnn.benchmark = True

    optimizer = optim.SGD(net.parameters(), lr=base_learning_rate, momentum=0.9, weight_decay=args.decay)
    lr_scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=200, eta_min=4e-08)

    for epoch in range(start_epoch, 200):
        print(epoch)
        print("====")
        # adjust_learning_rate(optimizer, epoch)
        # train_loss, train_acc = train_with_exhaustive_testing(epoch)
        train_loss, train_acc = hyper_train(epoch)
        # train_loss, train_acc = train(epoch)
        test_loss, test_acc = test(epoch)

        lr_scheduler.step()
        with open(logname, 'a') as logfile:
            logwriter = csv.writer(logfile, delimiter=',')
            logwriter.writerow([epoch, train_loss, train_acc, test_loss, test_acc])





