import MBD
import numpy as np
import time
from PIL import Image
import matplotlib.pyplot as plt





def demo_MBD_distance2d_gray_scale_image():
    I = Image.open('/media/minh/MEDIA/Study/deeplearning/Dahunet/DAHU-Net/BDCN_backbone/ISBI_dataset/ISBI13/contour_image/8.png').convert('L')
    I = np.array(I, dtype = np.uint8)
    

    seed_pos = [436, 444]
    S = np.zeros((I.shape[0], I.shape[1]), np.uint8)
    S[seed_pos[0]][seed_pos[1]] = 1
    D1 = MBD.MBD_waterflow(I,S)
    plt.imshow(D1)
    plt.show()


if __name__ == '__main__':

    demo_MBD_distance2d_gray_scale_image()

