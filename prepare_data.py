import numpy as np
import os
import PIL
import PIL.Image
import pathlib



folder = '/media/minh/MEDIA/Study/deeplearning/interactive_segmentation/dahu/Inside-Outside-Guidance/datasets/VOCdevkit/VOC2012/JPEGImages/'



files = os.listdir(folder)

N = len(files)
K = 0.2 * N
arr = np.array([0] * int(K) + [1] * int(N-K))

np.random.shuffle(arr)

train = []
val = []
for i in range(0, N):
    parts = files[i].split('.')
    if arr[i]:
        train.append(parts[0])
    else:
        val.append(parts[0])


with open('/media/minh/MEDIA/Study/deeplearning/interactive_segmentation/iog/dahu/datasets/VOCdevkit/VOC2012/ImageSets/prepare/train.txt', 'w') as f:
    for item in train:
        f.write("%s\n" % item)

with open('/media/minh/MEDIA/Study/deeplearning/interactive_segmentation/iog/dahu/datasets/VOCdevkit/VOC2012/ImageSets/prepare/val.txt', 'w') as f:
    for item in val:
        f.write("%s\n" % item)