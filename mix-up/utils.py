"""
helper functions:
- get_mean_and_std: calculate the mean and std value of dataset.
- msr_init: net parameter initialization.
- progress_bar: progress bar mimic xlua.progress
"""

import os
import sys
import time
import math

import torch.nn as nn
import torch.nn.init as init

import numpy as np
import torch


"""
multi page test functions
"""

def mul_mixup_data(x, y, alpha=1.0, use_cuda=True, sub_data_num=4):
    '''Compute the mixup data. Return mixed inputs, pairs of targets, and lambda'''
    if alpha > 0.:
        lam = np.random.beta(alpha, alpha)
    else:
        lam = 1.
    print(lam)
    batch_size = x.size()[0]
    sub_index = []
    mixed_x = lam * x
    sub_weight = (1-lam)/sub_data_num
    for _ in range(sub_data_num):
        if use_cuda:
            index = torch.randperm(batch_size).cuda()
        else:
            index = torch.randperm(batch_size)
        mixed_x = mixed_x + (1 - lam) * x[index, :]
        sub_index.append(y[index].reshape(1,batch_size))
    return mixed_x, y, torch.cat(sub_index), lam


def mul_mixup_criterion(y, sub_y, lam):
    return lambda criterion, pred: lam * criterion(pred, y) + (1 - lam) * sum([criterion(pred, x) for x in sub_y]) / len(sub_y)




"""
origin 2 page test
"""

def mixup_data(x, y, alpha=1.0, use_cuda=True):

    if alpha > 0.:
        lam = np.random.beta(alpha, alpha)
    else:
        lam = 1.
    lam = 0.8
    batch_size = x.size()[0]
    if use_cuda:
        index = torch.randperm(batch_size).cuda()
    else:
        index = torch.randperm(batch_size)

    mixed_x = lam * x + (1 - lam) * x[index,:]
    y_a, y_b = y, y[index]
    return mixed_x, y_a, y_b, lam


def mixup_criterion(y_a, y_b, lam):
    return lambda criterion, pred: torch.mean(lam * criterion(pred, y_a)) + torch.mean((1 - lam) * criterion(pred, y_b))



"""
exhaustive testing functions
"""

def exhaustive_mix_data_pre(x, y, index, use_cuda=True):
    # lam = np.random.beta(1, 1)
    lam = 0.8

    batch_size = x.size()[0]
    index_data = torch.full([batch_size], index)
    if use_cuda:
        index_data = index_data.long()
    # if index > batch_size:
    #     index_data = torch.randperm(batch_size).cuda()
    mixed_x = lam * x[index_data, :] + (1 - lam) * x
    y_a, y_b = y[index_data], y
    return mixed_x, y_a, y_b, lam



def exhausitive_mix_data(x, x_pair, y, lam):
    # lam = np.random.beta(1, 1)
    #
    # mix_data = lam * x + (1 - lam) * x[x_pair,:]
    # return mix_data, y, y[x_pair], lam
    batch_size = x.size()[0]

    x = x.reshape(batch_size, -1)
    lam = lam.reshape(batch_size, -1)

    mix_data = lam * x + (1 - lam).cuda() * x[x_pair, :]
    mix_data = mix_data.reshape(batch_size, 3, 32, 32)
    return mix_data, y, y[x_pair], lam


"""
hyper search
"""



def hyper_mixup_criterion(y):
    return lambda criterion, pred, hyper:  criterion(pred, y)


"""
support functions
"""


def get_mean_and_std(dataset):
    dataloader = torch.utils.data.DataLoader(dataset, batch_size=1, shuffle=True, num_workers=2)
    mean = torch.zeros(3)
    std = torch.zeros(3)
    print('==> Computing mean and std..')
    for inputs, targets in dataloader:
        for i in range(3):
            mean[i] += inputs[:,i,:,:].mean()
            std[i] += inputs[:,i,:,:].std()
    mean.div_(len(dataset))
    std.div_(len(dataset))
    return mean, std

def init_params(net):
    for m in net.modules():
        if isinstance(m, nn.Conv2d):
            init.kaiming_normal(m.weight, mode='fan_out')
            if m.bias:
                init.constant(m.bias, 0)
        elif isinstance(m, nn.BatchNorm2d):
            init.constant(m.weight, 1)
            init.constant(m.bias, 0)
        elif isinstance(m, nn.Linear):
            init.normal(m.weight, std=1e-3)
            if m.bias:
                init.constant(m.bias, 0)


# _, term_width = os.popen('stty size', 'r').read().split()
term_width = 20
term_width = int(term_width)

TOTAL_BAR_LENGTH = 65.
last_time = time.time()
begin_time = last_time

def progress_bar(current, total, msg=None):
    global last_time, begin_time
    if current == 0:
        begin_time = time.time()  # Reset for new bar.

    cur_len = int(TOTAL_BAR_LENGTH*current/total)
    rest_len = int(TOTAL_BAR_LENGTH - cur_len) - 1

    sys.stdout.write(' [')
    for i in range(cur_len):
        sys.stdout.write('=')
    sys.stdout.write('>')
    for i in range(rest_len):
        sys.stdout.write('.')
    sys.stdout.write(']')

    cur_time = time.time()
    step_time = cur_time - last_time
    last_time = cur_time
    tot_time = cur_time - begin_time

    L = []
    L.append('  Step: %s' % format_time(step_time))
    L.append(' | Tot: %s' % format_time(tot_time))
    if msg:
        L.append(' | ' + msg)

    msg = ''.join(L)
    sys.stdout.write(msg)
    for i in range(term_width-int(TOTAL_BAR_LENGTH)-len(msg)-3):
        sys.stdout.write(' ')

    # Go back to the center of the bar.
    for i in range(term_width-int(TOTAL_BAR_LENGTH/2)+2):
        sys.stdout.write('\b')
    sys.stdout.write(' %d/%d ' % (current+1, total))

    if current < total-1:
        sys.stdout.write('\r')
    else:
        sys.stdout.write('\n')
    sys.stdout.flush()

def format_time(seconds):
    days = int(seconds / 3600/24)
    seconds = seconds - days*3600*24
    hours = int(seconds / 3600)
    seconds = seconds - hours*3600
    minutes = int(seconds / 60)
    seconds = seconds - minutes*60
    secondsf = int(seconds)
    seconds = seconds - secondsf
    millis = int(seconds*1000)

    f = ''
    i = 1
    if days > 0:
        f += str(days) + 'D'
        i += 1
    if hours > 0 and i <= 2:
        f += str(hours) + 'h'
        i += 1
    if minutes > 0 and i <= 2:
        f += str(minutes) + 'm'
        i += 1
    if secondsf > 0 and i <= 2:
        f += str(secondsf) + 's'
        i += 1
    if millis > 0 and i <= 2:
        f += str(millis) + 'ms'
        i += 1
    if f == '':
        f = '0ms'
    return f


