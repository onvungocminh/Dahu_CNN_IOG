import os.path

from torch.utils.data import DataLoader
from evaluation.eval import eval_one_result
import dataloaders.pascal as pascal
from PIL import Image
import cv2
import numpy as np
import matplotlib.pyplot as plt


exp_root_dir = './'

method_names = []
method_names.append('run')

if __name__ == '__main__':

    # Dataloader

    input_folder = '/media/minh/MEDIA/Study/deeplearning/interactive_segmentation/iog/dahu/run/Results/'
    gt_folder = '/media/minh/MEDIA/Study/deeplearning/interactive_segmentation/iog/dahu/run/GT/'

    # Iterate through all the different methods

    files = os.listdir(input_folder)

    IoU_total = 0
    for file in files:
        print(file)

        # parts = file.split('.')
        # part = parts[0]

        img_file = input_folder + file
        gt_file = gt_folder + file

        img = cv2.imread(img_file)
        img_gray =  cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)


        gt_img = cv2.imread(gt_file)
        gt_gray =  cv2.cvtColor(gt_img, cv2.COLOR_BGR2GRAY)
        gt_gray = gt_gray /255


        ret, thresh = cv2.threshold(img_gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)

        thresh = thresh / 255

        # plt.imshow(thresh)
        # plt.show()

        A = thresh +gt_gray
        B = (A == 2)
        C = (A>0)
        inter = np.sum(B)
        union = np.sum(C)
        IoU = inter/(union+ 0.0001)
        IoU_total = IoU_total + IoU

    print("IoU: ", IoU_total/len(files))












