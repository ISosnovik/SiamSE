"""
MIT License

Copyright (c) 2021 Ivan Sosnovik, Artem Moskalev
Copyright (c) 2018 Microsoft (Houwen Peng, Zhipeng Zhang)
"""

import os
import cv2
import argparse
import numpy as np
import yaml

import lib.models.models as models
from lib.tracker import SESiamFCTracker
from lib.utils import load_pretrain, cxy_wh_2_rect, get_axis_aligned_bbox, load_dataset, poly_iou, convert_color_RGB


def track(tracker, video, dataset_name):
    start_frame, toc = 0, 0

    tracker_path = os.path.join('result', dataset_name, 'SESiamFCTracker')
    if not os.path.exists(tracker_path):
        os.makedirs(tracker_path)

    if 'VOT' in dataset_name:
        baseline_path = os.path.join(tracker_path, 'baseline')
        video_path = os.path.join(baseline_path, video['name'])
        if not os.path.exists(video_path):
            os.makedirs(video_path)
        result_path = os.path.join(video_path, video['name'] + '_001.txt')
    else:
        result_path = os.path.join(tracker_path, '{:s}.txt'.format(video['name']))

    if os.path.exists(result_path):
        return

    regions = []
    image_files, gt = video['image_files'], video['gt']

    for f, image_file in enumerate(image_files):
        im = cv2.imread(image_file)
        im = convert_color_RGB(im)

        tic = cv2.getTickCount()

        if f == start_frame:  # init
            cx, cy, w, h = get_axis_aligned_bbox(gt[f])
            target_pos = np.array([cx, cy])
            target_sz = np.array([w, h])
            tracker.init(im, target_pos, target_sz)  # init tracker
            regions.append(1 if 'VOT' in dataset_name else gt[f])

        elif f > start_frame:  # tracking
            target_pos, target_sz = tracker.track(im)
            location = cxy_wh_2_rect(target_pos, target_sz)
            b_overlap = poly_iou(gt[f], location) if 'VOT' in dataset_name else 1
            if b_overlap > 0:
                regions.append(location)
            else:
                regions.append(2)
                start_frame = f + 5
        else:
            regions.append(0)

        toc += cv2.getTickCount() - tic

    with open(result_path, "w") as fin:
        if 'VOT' in dataset_name:
            for x in regions:
                if isinstance(x, int):
                    fin.write("{:d}\n".format(x))
                else:
                    p_bbox = x.copy()
                    fin.write(','.join([str(i) for i in p_bbox]) + '\n')
        else:
            for x in regions:
                p_bbox = x.copy()
                fin.write(
                    ','.join([str(i + 1) if idx == 0 or idx == 1 else str(i) for idx, i in enumerate(p_bbox)]) + '\n')

    toc /= cv2.getTickFrequency()
    print('Video: {:12s} Time: {:2.1f}s Speed: {:3.1f}fps'.format(video['name'], toc, f / toc))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--checkpoint', required=True, type=str, help='pretrained model')
    parser.add_argument('--dataset', default='VOT2017', help='dataset test')
    parser.add_argument('--dataset_root', default=None)
    parser.add_argument('--cfg', required=True)
    args = parser.parse_args()

    with open(args.cfg) as f:
        tracker_config = yaml.load(f.read())

    # prepare model
    net = models.__dict__[tracker_config['MODEL']](padding_mode='constant')
    net = load_pretrain(net, args.checkpoint)
    net = net.eval().cuda()

    # prepare tracker
    tracker_config = tracker_config['TRACKER'][args.dataset]
    tracker = SESiamFCTracker(net, **tracker_config)
    print('Tracker')
    print(tracker)

    # prepare video
    dataset = load_dataset(args.dataset, root=args.dataset_root)
    video_keys = list(dataset.keys()).copy()

    # tracking all videos in benchmark
    for video in video_keys:
        track(tracker, dataset[video], dataset_name=args.dataset)


if __name__ == '__main__':
    main()
