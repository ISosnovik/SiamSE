"""
MIT License

Copyright (c) 2021 Ivan Sosnovik, Artem Moskalev
Copyright (c) 2018 Microsoft (Houwen Peng, Zhipeng Zhang)
"""

from __future__ import division

import os
import cv2
import json
import random
import numpy as np
import torchvision.transforms as transforms
from torch.utils.data import Dataset

from .utils import *

sample_random = random.Random()


class SiamFCDataset(Dataset):
    def __init__(self, cfg):
        super(SiamFCDataset, self).__init__()
        # pair information
        self.template_size = cfg.SIAMFC.TRAIN.TEMPLATE_SIZE
        self.search_size = cfg.SIAMFC.TRAIN.SEARCH_SIZE
        self.stride = cfg.SIAMFC.TRAIN.STRIDE
        self.size = (self.search_size - self.template_size) // self.stride + 1

        # aug information
        self.shift = cfg.SIAMFC.DATASET.SHIFT
        self.scale = cfg.SIAMFC.DATASET.SCALE
        self.anno = cfg.SIAMFC.DATASET.GOT10K.ANNOTATION
        self.num_use = cfg.SIAMFC.TRAIN.PAIRS
        self.root = cfg.SIAMFC.DATASET.GOT10K.PATH

        self.transform_extra = transforms.Compose([
            transforms.ToPILImage(),
            transforms.ColorJitter(0.05, 0.05, 0.05, 0.05)
        ])

        self.labels = json.load(open(self.anno, 'r'))
        self.videos = list(self.labels.keys())
        self.num = len(self.videos)
        self.frame_range = 100
        self.pick = self._shuffle()

    def __len__(self):
        return self.num_use

    def __getitem__(self, index):
        index = self.pick[index]
        template, search = self._get_pairs(index)

        template_image = cv2.imread(template[0])
        template_image = convert_color_RGB(template_image)

        search_image = cv2.imread(search[0])
        search_image = convert_color_RGB(search_image)

        template_box = self._toBBox(template_image, template[1])
        search_box = self._toBBox(search_image, search[1])

        template, _, _ = self._augmentation(template_image, template_box, self.template_size)
        search, bbox, dag_param = self._augmentation(search_image, search_box, self.search_size)

        # from PIL image to numpy
        template = np.array(template).transpose((2, 0, 1)).astype('float32')
        search = np.array(search).transpose((2, 0, 1)).astype('float32')

        out_label = self._dynamic_label(self.size, dag_param['shift'])
        return template, search, out_label, np.array(bbox, np.float32)

    def _shuffle(self):
        lists = list(range(0, self.num))
        m = 0
        pick = []
        while m < self.num_use:
            sample_random.shuffle(lists)
            pick += lists
            m += self.num

        self.pick = pick[:self.num_use]
        return self.pick

    def _get_image_anno(self, video, track, frame):
        frame = "{:06d}".format(frame)
        image_path = os.path.join(self.root, video, "{}.{}.x.jpg".format(frame, track))
        image_anno = self.labels[video][track][frame]
        return image_path, image_anno

    def _get_pairs(self, index):
        video_name = self.videos[index]
        video = self.labels[video_name]
        track = random.choice(list(video.keys()))
        track_info = video[track]
        try:
            frames = track_info['frames']
        except:
            frames = list(track_info.keys())

        template_frame = random.randint(0, len(frames) - 1)

        left = max(template_frame - self.frame_range, 0)
        right = min(template_frame + self.frame_range, len(frames) - 1) + 1
        search_range = frames[left:right]
        template_frame = int(frames[template_frame])
        search_frame = int(random.choice(search_range))
        anno_t = self._get_image_anno(video_name, track, template_frame)
        anno_s = self._get_image_anno(video_name, track, search_frame)
        return anno_t, anno_s

    def _posNegRandom(self):
        return random.random() * 2 - 1.0

    def _toBBox(self, image, shape):
        imh, imw = image.shape[:2]
        if len(shape) == 4:
            w, h = shape[2] - shape[0], shape[3] - shape[1]
        else:
            w, h = shape
        context_amount = 0.5
        exemplar_size = self.template_size
        wc_z = w + context_amount * (w + h)
        hc_z = h + context_amount * (w + h)
        s_z = np.sqrt(wc_z * hc_z)
        scale_z = exemplar_size / s_z
        w = w * scale_z
        h = h * scale_z
        cx, cy = imw // 2, imh // 2
        bbox = center2corner(Center(cx, cy, w, h))
        return bbox

    def _crop_hwc(self, image, bbox, out_sz, padding=(0, 0, 0)):
        bbox = [float(x) for x in bbox]
        a = (out_sz - 1) / (bbox[2] - bbox[0])
        b = (out_sz - 1) / (bbox[3] - bbox[1])
        c = -a * bbox[0]
        d = -b * bbox[1]
        mapping = np.array([[a, 0, c],
                            [0, b, d]]).astype(np.float)
        crop = cv2.warpAffine(image, mapping, (out_sz, out_sz),
                              borderMode=cv2.BORDER_CONSTANT, borderValue=padding)
        return crop

    # ------------------------------------
    # function for data augmentation
    # ------------------------------------
    def _augmentation(self, image, bbox, size):
        shape = image.shape
        crop_bbox = center2corner((shape[0] // 2, shape[1] // 2, size, size))
        param = {}
        param['shift'] = (self._posNegRandom() * self.shift,
                          self._posNegRandom() * self.shift)   # shift
        param['scale'] = ((1.0 + self._posNegRandom() * self.scale),
                          (1.0 + self._posNegRandom() * self.scale))  # scale change

        crop_bbox, _ = aug_apply(Corner(*crop_bbox), param, shape)

        x1, y1 = crop_bbox.x1, crop_bbox.y1
        bbox = BBox(bbox.x1 - x1, bbox.y1 - y1, bbox.x2 - x1, bbox.y2 - y1)

        scale_x, scale_y = param['scale']
        bbox = Corner(bbox.x1 / scale_x, bbox.y1 / scale_y, bbox.x2 / scale_x, bbox.y2 / scale_y)

        image = self._crop_hwc(image, crop_bbox, size)   # shift and scale
        image = self.transform_extra(image)        # other data augmentation
        return image, bbox, param

    def _dynamic_label(self, label_size, c_shift, rPos=2):
        assert (label_size % 2 == 1)
        sz_x = label_size // 2 + round(-c_shift[0]) // self.stride
        sz_y = label_size // 2 + round(-c_shift[1]) // self.stride

        x, y = np.meshgrid(np.arange(0, label_size) - np.floor(float(sz_x)),
                           np.arange(0, label_size) - np.floor(float(sz_y)))

        dist_to_center = np.abs(x) + np.abs(y)  # Block metric
        label = np.where(dist_to_center <= rPos, 1.0, 0.0)
        return label
