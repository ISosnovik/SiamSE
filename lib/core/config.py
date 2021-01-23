"""
MIT License

Copyright (c) 2021 Ivan Sosnovik, Artem Moskalev
Copyright (c) 2018 Microsoft (Houwen Peng, Zhipeng Zhang)
"""
from yacs.config import CfgNode as CN

config = CN()
config.SIAMFC = CN()
config.SIAMFC.TRAIN = CN()
config.SIAMFC.DATASET = CN()
config.SIAMFC.DATASET.GOT10K = CN()

# general
config.WORKERS = 32
config.PRINT_FREQ = 10
config.CHECKPOINT_DIR = './snapshot'


# dataset
config.SIAMFC.DATASET.GOT10K.PATH = 'datasets/GOT10K/crop255'
config.SIAMFC.DATASET.GOT10K.ANNOTATION = 'datasets/GOT10K/train.json'
config.SIAMFC.DATASET.SHIFT = 4
config.SIAMFC.DATASET.SCALE = 0.05
config.SIAMFC.DATASET.COLOR = 1

# train
config.SIAMFC.TRAIN.MODEL = "SESiamFCResNet22"
config.SIAMFC.TRAIN.START_EPOCH = 0
config.SIAMFC.TRAIN.END_EPOCH = 50
config.SIAMFC.TRAIN.TEMPLATE_SIZE = 127
config.SIAMFC.TRAIN.SEARCH_SIZE = 255
config.SIAMFC.TRAIN.STRIDE = 8
config.SIAMFC.TRAIN.BATCH = 32
config.SIAMFC.TRAIN.PAIRS = 200000
config.SIAMFC.TRAIN.PRETRAIN = './pretrain/SESiamFCResNet22_weights.pth'
config.SIAMFC.TRAIN.LR = 0.001
config.SIAMFC.TRAIN.LR_END = 0.0000001
config.SIAMFC.TRAIN.MOMENTUM = 0.9
config.SIAMFC.TRAIN.WEIGHT_DECAY = 0.0001


def update_config(config_file):
    config.merge_from_file(config_file)
