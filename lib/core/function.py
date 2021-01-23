"""
MIT License

Copyright (c) 2021 Ivan Sosnovik, Artem Moskalev
Copyright (c) 2018 Microsoft (Houwen Peng, Zhipeng Zhang)
"""

import math
import time
import torch

from torch.autograd import Variable
from ..utils import print_speed


def siamfc_train(train_loader, model, optimizer, epoch, cur_lr, cfg):
    # unfix for FREEZE-OUT method
    model, optimizer = unfix_more(model, optimizer, epoch, cfg, cur_lr)

    # prepare
    batch_time = AverageMeter()
    data_time = AverageMeter()
    losses = AverageMeter()
    end = time.time()

    # switch to train mode
    model.train()
    model = model.cuda()

    for iter, input in enumerate(train_loader):
        # measure data loading time
        data_time.update(time.time() - end)

        # input and output/loss
        label_cls = input[2].type(torch.FloatTensor)  # BCE need float
        template = Variable(input[0]).cuda()
        search = Variable(input[1]).cuda()
        label_cls = Variable(label_cls).cuda()

        loss = model(template, search, label_cls)
        loss = torch.mean(loss)

        # compute gradient and do update step
        optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm(model.parameters(), 10)  # gradient clip

        if is_valid_number(loss.item()):
            optimizer.step()

        # record loss
        loss = loss.item()
        losses.update(loss, template.size(0))
        batch_time.update(time.time() - end)
        end = time.time()

        if (iter + 1) % cfg.PRINT_FREQ == 0:
            print('Epoch: [{0}][{1}/{2}] lr: {lr:.7f}\t Batch Time: {batch_time.avg:.3f}s \t Data Time:{data_time.avg:.3f}s \t Loss:{loss.avg:.5f}'.format(
                epoch, iter + 1, len(train_loader), lr=cur_lr, batch_time=batch_time, data_time=data_time, loss=losses))

            print_speed((epoch - 1) * len(train_loader) + iter + 1, batch_time.avg,
                        cfg.SIAMFC.TRAIN.END_EPOCH * len(train_loader))

    return model


def unfix_more(model, optimizer, epoch, cfg, cur_lr):
    if model.module.features.unfix(epoch / cfg.SIAMFC.TRAIN.END_EPOCH):
        print('unfix part model.')
        trainable_params = [p for p in model.parameters() if p.requires_grad]
        optimizer = torch.optim.SGD(trainable_params, cur_lr,
                                    momentum=cfg.SIAMFC.TRAIN.MOMENTUM,
                                    weight_decay=cfg.SIAMFC.TRAIN.WEIGHT_DECAY)

        print('trainable params:')
        for name, param in model.named_parameters():
            if param.requires_grad:
                print(name)
    else:
        print('no unfix in this epoch.')

    return model, optimizer


def is_valid_number(x):
    return not(math.isnan(x) or math.isinf(x) or x > 1e4)


class AverageMeter(object):
    """Computes and stores the average and current value"""

    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count if self.count != 0 else 0
