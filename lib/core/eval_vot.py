"""
MIT License

Copyright (c) 2021 Ivan Sosnovik, Artem Moskalev
Copyright (c) 2018 Microsoft (Houwen Peng, Zhipeng Zhang)
"""


import sys
import os

absPath = os.path.abspath('.')
eval_path = os.path.join(absPath, 'lib/core/')

try:
    import matlab.engine  # causes error for pytorch version < 1.5
except:
    print('Error importing matlab.engine')

try:
    eng = matlab.engine.start_matlab()  # for test eao in vot-toolkit
    eng.cd('./lib/core')
except:
    print('Error starting matlab.engine')


def eval_vot(dataset, result_path, tracker_reg):

    #trackers = listdir(join(result_path, dataset))

    # for tracker in trackers:
    base_path = os.path.join(result_path, dataset, tracker_reg, 'baseline')
    #print('base_path:', base_path)

    eao = eval_eao(base_path, dataset)

    print('[*] tracker: {0} : EAO: {1}'.format(tracker_reg, eao))
    eng.cd(eval_path)


def eval_eao(base_path, dataset):
    """
    start matlab engin and test eao in vot toolkit
    """
    results = []
    videos = sorted(os.listdir(base_path))  # must sorted!!

    for video in videos:
        video_re = []
        path_v = os.path.join(base_path, video, '{}_001.txt'.format(video))
        fin = open(path_v).readlines()

        for line in fin:
            line = eval(line)  # tuple
            if isinstance(line, float) or isinstance(line, int):
                line = [float(line)]   # have to be float
            else:
                line = list(line)
            video_re.append(line)
        results.append(video_re)

    year = dataset.split('VOT')[-1]
    eao = eng.get_eao(results, year)

    return eao


if __name__ == '__main__':
    if len(sys.argv) < 2:
        print('python ./lib/core/eval_vot.py VOT2017 ./result')
        exit()
    dataset = sys.argv[1]
    result_path = sys.argv[2]
    tracker_reg = sys.argv[3]
    eval_vot(dataset, result_path, tracker_reg)
