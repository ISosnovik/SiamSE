"""
MIT License

Copyright (c) 2021 Ivan Sosnovik, Artem Moskalev
Copyright (c) 2018 Microsoft (Houwen Peng, Zhipeng Zhang)
"""

import math
import pprint
import argparse
import numpy as np

import torch
from torch.utils.data import DataLoader

import lib.models.models as models
from lib.utils import print_speed, load_pretrain, save_model
from lib.dataset import SiamFCDataset
from lib.core.config import config, update_config
from lib.core.function import siamfc_train


parser = argparse.ArgumentParser()
parser.add_argument('--cfg', required=True, type=str, help='yaml configure file name')
args = parser.parse_args()
update_config(args.cfg)


print('Config:')
print(pprint.pformat(config))
print()

model = models.__dict__[config.SIAMFC.TRAIN.MODEL]()
model = load_pretrain(model, config.SIAMFC.TRAIN.PRETRAIN)

trainable_params = [p for p in model.parameters() if p.requires_grad]

optimizer = torch.optim.SGD(trainable_params, config.SIAMFC.TRAIN.LR,
                            momentum=config.SIAMFC.TRAIN.MOMENTUM,
                            weight_decay=config.SIAMFC.TRAIN.WEIGHT_DECAY)

lr_scheduler = np.logspace(math.log10(config.SIAMFC.TRAIN.LR),
                           math.log10(config.SIAMFC.TRAIN.LR_END),
                           config.SIAMFC.TRAIN.END_EPOCH)

gpu_num = torch.cuda.device_count()
model = torch.nn.DataParallel(model, device_ids=range(gpu_num)).cuda()
print('Model is using {} GPU(s)'.format(gpu_num))

for epoch in range(config.SIAMFC.TRAIN.START_EPOCH, config.SIAMFC.TRAIN.END_EPOCH):
    train_set = SiamFCDataset(config)
    train_loader = DataLoader(train_set, batch_size=config.SIAMFC.TRAIN.BATCH * gpu_num,
                              num_workers=config.WORKERS, pin_memory=True, sampler=None)

    cur_lr = lr_scheduler[epoch]
    for param_group in optimizer.param_groups:
        param_group['lr'] = cur_lr

    model = siamfc_train(train_loader, model=model, optimizer=optimizer,
                         epoch=epoch + 1, cur_lr=cur_lr, cfg=config)

    if epoch >= 4:
        save_model(model, epoch, optimizer, config.SIAMFC.TRAIN.MODEL, config)
