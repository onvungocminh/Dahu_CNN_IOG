import os
import torch, cv2
import random
import numpy as np
import matplotlib.pyplot as plt
from scipy import ndimage
import appro_dahu_3chanels as dahu
import math
from skimage.util import img_as_float
import skimage
import scipy.spatial.distance
from skimage.color import rgb2lab
from sklearn import mixture
import MBD

def tens2image(im):
    if im.size()[0] == 1:
        tmp = np.squeeze(im.numpy(), axis=0)
    else:
        tmp = im.numpy()
    if tmp.ndim == 2:
        return tmp
    else:
        return tmp.transpose((1, 2, 0))


def crop2fullmask(crop_mask, bbox, im=None, im_size=None, zero_pad=False, relax=0, mask_relax=True,
                  interpolation=cv2.INTER_CUBIC, scikit=False):
    if scikit:
        from skimage.transform import resize as sk_resize
    assert(not(im is None and im_size is None)), 'You have to provide an image or the image size  True'  
    if im is None:
        im_si = im_size
    else:
        im_si = im.shape
    # Borers of image
    bounds = (0, 0, im_si[1] - 1, im_si[0] - 1)

    # Valid bounding box locations as (x_min, y_min, x_max, y_max)
    bbox_valid = (max(bbox[0], bounds[0]),
                  max(bbox[1], bounds[1]),
                  min(bbox[2], bounds[2]),
                  min(bbox[3], bounds[3]))

    # Bounding box of initial mask
    bbox_init = (bbox[0] + relax,
                 bbox[1] + relax,
                 bbox[2] - relax,
                 bbox[3] - relax)

    if zero_pad:
        # Offsets for x and y
        offsets = (-bbox[0], -bbox[1])
    else:
#        assert((bbox == bbox_valid).all())
        offsets = (-bbox_valid[0], -bbox_valid[1])

    # Simple per element addition in the tuple
    inds = tuple(map(sum, zip(bbox_valid, offsets + offsets)))

    if scikit:
        crop_mask = sk_resize(crop_mask, (bbox[3] - bbox[1] + 1, bbox[2] - bbox[0] + 1), order=0, mode='constant').astype(crop_mask.dtype)
    else:
        crop_mask = cv2.resize(crop_mask, (bbox[2] - bbox[0] + 1, bbox[3] - bbox[1] + 1), interpolation=interpolation)
    result_ = np.zeros(im_si)
    result_[bbox_valid[1]:bbox_valid[3] + 1, bbox_valid[0]:bbox_valid[2] + 1] = \
        crop_mask[inds[1]:inds[3] + 1, inds[0]:inds[2] + 1]

    result = np.zeros(im_si)
    if mask_relax:
        result[bbox_init[1]:bbox_init[3]+1, bbox_init[0]:bbox_init[2]+1] = \
            result_[bbox_init[1]:bbox_init[3]+1, bbox_init[0]:bbox_init[2]+1]
    else:
        result = result_

    return result


def overlay_mask(im, ma, colors=None, alpha=0.5):
    assert np.max(im) <= 1.0
    if colors is None:
        colors = np.load(os.path.join(os.path.dirname(__file__), 'pascal_map.npy'))/255.
    else:
        colors = np.append([[0.,0.,0.]], colors, axis=0);

    if ma.ndim == 3:
        assert len(colors) >= ma.shape[0], 'Not enough colors'
    ma = ma.astype(np.bool)
    im = im.astype(np.float32)

    if ma.ndim == 2:
        fg = im * alpha+np.ones(im.shape) * (1 - alpha) * colors[1, :3]   # np.array([0,0,255])/255.0
    else:
        fg = []
        for n in range(ma.ndim):
            fg.append(im * alpha + np.ones(im.shape) * (1 - alpha) * colors[1+n, :3])
    # Whiten background
    bg = im.copy()
    if ma.ndim == 2:
        bg[ma == 0] = im[ma == 0]
        bg[ma == 1] = fg[ma == 1]
        total_ma = ma
    else:
        total_ma = np.zeros([ma.shape[1], ma.shape[2]])
        for n in range(ma.shape[0]):
            tmp_ma = ma[n, :, :]
            total_ma = np.logical_or(tmp_ma, total_ma)
            tmp_fg = fg[n]
            bg[tmp_ma == 1] = tmp_fg[tmp_ma == 1]
        bg[total_ma == 0] = im[total_ma == 0]

    # [-2:] is s trick to be compatible both with opencv 2 and 3
    contours = cv2.findContours(total_ma.copy().astype(np.uint8), cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)[-2:]
    cv2.drawContours(bg, contours[0], -1, (0.0, 0.0, 0.0), 1)

    return bg
import PIL
def overlay_masks(im, masks, alpha=0.5):
    colors = np.load(os.path.join(os.path.dirname(__file__), 'pascal_map.npy'))/255.
    
    if isinstance(masks, np.ndarray):
        masks = [masks]

    assert len(colors) >= len(masks), 'Not enough colors'

    ov = im.copy()
    ov_black = im.copy()*0
    
    imgZero = np.zeros(np.array(masks, dtype = np.uint8).shape,np.uint8)
    im = im.astype(np.float32)
    total_ma = np.zeros([im.shape[0], im.shape[1]])
    i = 1
    for ma in masks:
        ma = ma.astype(np.bool)
        fg = im * alpha+np.ones(im.shape) * (1 - alpha) * colors[i, :3]   # np.array([0,0,255])/255.0
        i = i + 1
        ov[ma == 1] = fg[ma == 1]
        total_ma += ma

        # [-2:] is s trick to be compatible both with opencv 2 and 3
        contours = cv2.findContours(ma.copy().astype(np.uint8), cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)[-2:]
        cv2.drawContours(ov, contours[0], -1, (0.0, 0.0, 0.0), 1)
        cv2.drawContours(ov_black, contours[0], -1, (255, 255, 255), -1)#only draw a round
    ov[total_ma == 0] = im[total_ma == 0]

    return ov_black

from scipy import ndimage    
def getPositon(distance_transform):
    a = np.mat(distance_transform) 
    raw, column = a.shape# get the matrix of a raw and column
    _positon = np.argmax(a)# get the index of max in the a
    m, n = divmod(_positon, column)
    raw=m
    column=n
#    print "The raw is " ,m
#    print "The column is ",  n
#    print "The max of the a is ", a[m , n]
#    print(raw,column,a[m , n])
    return  raw,column

def iog_points(mask, pad_pixel=10):
    def find_point(id_x, id_y, ids):
        sel_id = ids[0][random.randint(0, len(ids[0]) - 1)]
        return [id_x[sel_id], id_y[sel_id]]

    inds_y, inds_x = np.where(mask > 0.5)   
    [h,w]=mask.shape
    left = find_point(inds_x, inds_y, np.where(inds_x <= np.min(inds_x))) 
    right = find_point(inds_x, inds_y, np.where(inds_x >= np.max(inds_x))) 
    top = find_point(inds_x, inds_y, np.where(inds_y <= np.min(inds_y))) 
    bottom = find_point(inds_x, inds_y, np.where(inds_y >= np.max(inds_y)))

    x_min=left[0]
    x_max=right[0]
    y_min=top[1]
    y_max=bottom[1]

    map_xor = (mask > 0.5)
    h,w = map_xor.shape
    map_xor_new = np.zeros((h+2,w+2))
    map_xor_new[1:(h+1),1:(w+1)] = map_xor[:,:]
    distance_transform=ndimage.distance_transform_edt(map_xor_new)
    distance_transform_back = distance_transform[1:(h+1),1:(w+1)]    
    raw,column=getPositon(distance_transform_back)
    center_point = [column,raw]

    left_top=[max(x_min-pad_pixel,0),  max(y_min-pad_pixel,0)]
    left_bottom=[max(x_min-pad_pixel ,0),     min(y_max+pad_pixel,h)]
    right_top=[min(x_max+pad_pixel,w),          max(y_min-pad_pixel,0)]
    righr_bottom=[min(x_max+pad_pixel ,w),    min(y_max+pad_pixel,h)]
    a=[center_point,left_top,left_bottom,right_top,righr_bottom]  

    return np.array(a)
    

def get_bbox(mask, points=None, pad=0, zero_pad=False):
    if points is not None:
        inds = np.flip(points.transpose(), axis=0)
    else:
        inds = np.where(mask > 0)

    if inds[0].shape[0] == 0:
        return None

    if zero_pad:
        x_min_bound = -np.inf
        y_min_bound = -np.inf
        x_max_bound = np.inf
        y_max_bound = np.inf
    else:
        x_min_bound = 0
        y_min_bound = 0
        x_max_bound = mask.shape[1] - 1
        y_max_bound = mask.shape[0] - 1

    x_min = max(inds[1].min() - pad, x_min_bound)
    y_min = max(inds[0].min() - pad, y_min_bound)
    x_max = min(inds[1].max() + pad, x_max_bound)
    y_max = min(inds[0].max() + pad, y_max_bound)

    return x_min, y_min, x_max, y_max


def crop_from_bbox(img, bbox, zero_pad=False):
    # Borders of image
    bounds = (0, 0, img.shape[1] - 1, img.shape[0] - 1)

    # Valid bounding box locations as (x_min, y_min, x_max, y_max)
    bbox_valid = (max(bbox[0], bounds[0]),
                  max(bbox[1], bounds[1]),
                  min(bbox[2], bounds[2]),
                  min(bbox[3], bounds[3]))

    if zero_pad:
        # Initialize crop size (first 2 dimensions)
        crop = np.zeros((bbox[3] - bbox[1] + 1, bbox[2] - bbox[0] + 1), dtype=img.dtype)

        # Offsets for x and y
        offsets = (-bbox[0], -bbox[1])

    else:
        assert(bbox == bbox_valid)
        crop = np.zeros((bbox_valid[3] - bbox_valid[1] + 1, bbox_valid[2] - bbox_valid[0] + 1), dtype=img.dtype)
        offsets = (-bbox_valid[0], -bbox_valid[1])

    # Simple per element addition in the tuple
    inds = tuple(map(sum, zip(bbox_valid, offsets + offsets)))

    img = np.squeeze(img)
    if img.ndim == 2:
        crop[inds[1]:inds[3] + 1, inds[0]:inds[2] + 1] = \
            img[bbox_valid[1]:bbox_valid[3] + 1, bbox_valid[0]:bbox_valid[2] + 1]
    else:
        crop = np.tile(crop[:, :, np.newaxis], [1, 1, 3])  # Add 3 RGB Channels
        crop[inds[1]:inds[3] + 1, inds[0]:inds[2] + 1, :] = \
            img[bbox_valid[1]:bbox_valid[3] + 1, bbox_valid[0]:bbox_valid[2] + 1, :]

    return crop


def fixed_resize(sample, resolution, flagval=None):

    if flagval is None:
        if ((sample == 0) | (sample == 1)).all():
            flagval = cv2.INTER_NEAREST
        else:
            flagval = cv2.INTER_CUBIC

    if isinstance(resolution, int):
        tmp = [resolution, resolution]
        tmp[np.argmax(sample.shape[:2])] = int(round(float(resolution)/np.min(sample.shape[:2])*np.max(sample.shape[:2])))
        resolution = tuple(tmp)

    if sample.ndim == 2 or (sample.ndim == 3 and sample.shape[2] == 3):
        sample = cv2.resize(sample, resolution[::-1], interpolation=flagval)
    else:
        tmp = sample
        sample = np.zeros(np.append(resolution, tmp.shape[2]), dtype=np.float32)
        for ii in range(sample.shape[2]):
            sample[:, :, ii] = cv2.resize(tmp[:, :, ii], resolution[::-1], interpolation=flagval)
    return sample


def crop_from_mask(img, mask, relax=0, zero_pad=False):
    if mask.shape[:2] != img.shape[:2]:
        mask = cv2.resize(mask, dsize=tuple(reversed(img.shape[:2])), interpolation=cv2.INTER_NEAREST)

    assert(mask.shape[:2] == img.shape[:2])
    bbox = get_bbox(mask, pad=relax, zero_pad=zero_pad)

    if bbox is None:
        return None

    crop = crop_from_bbox(img, bbox, zero_pad)

    return crop


def make_gaussian(size, sigma=10, center=None, d_type=np.float64):
    """ Make a square gaussian kernel.
    size: is the dimensions of the output gaussian
    sigma: is full-width-half-maximum, which
    can be thought of as an effective radius.
    """

    x = np.arange(0, size[1], 1, float)
    y = np.arange(0, size[0], 1, float)
    y = y[:, np.newaxis]

    if center is None:
        x0 = y0 = size[0] // 2
    else:
        x0 = center[0]
        y0 = center[1]

    return np.exp(-4 * np.log(2) * ((x - x0) ** 2 + (y - y0) ** 2) / sigma ** 2).astype(d_type)

    
def make_gt(image, img, labels, sigma=10, one_mask_per_point=False):
    """ Make the ground-truth for  landmark.
    img: the original color image
    labels: label with the Gaussian center(s) [[x0, y0],[x1, y1],...]
    sigma: sigma of the Gaussian.
    one_mask_per_point: masks for each point in different channels?
    """

    h, w = img.shape[:2]
    if labels is None:
        gt = make_gaussian((h, w), center=(h//2, w//2), sigma=sigma)
        gt_0 = np.zeros(shape=(h, w), dtype=np.float64)
        gt_0 = gt
        gt_1 = np.zeros(shape=(h, w), dtype=np.float64)
        gt_2 = np.zeros(shape=(h, w), dtype=np.float64)

        gtout = np.zeros(shape=(h, w, 3))
        gtout[:, :, 0]=gt_0
        gtout[:, :, 1]=gt_1
        gtout[:, :, 2]=gt_2


        gtout = gtout.astype(dtype=img.dtype) #(0~1)        
        return gtout
    else:
        labels = np.array(labels)
        if labels.ndim == 1:
            labels = labels[np.newaxis]       
            gt_0 = np.zeros(shape=(h, w), dtype=np.float64)
            gt_1 = np.zeros(shape=(h, w), dtype=np.float64)
            gt_2 = np.zeros(shape=(h, w), dtype=np.float64)


            gt_0 = np.maximum(gt_0, make_gaussian((h, w), center=labels[0, :], sigma=sigma))
           
        else:   
            gt_0 = np.zeros(shape=(h, w), dtype=np.float64)
            gt_1 = np.zeros(shape=(h, w), dtype=np.float64)
            gt_2 = np.zeros(shape=(h, w), dtype=np.float64)


            # for ii in range(1,labels.shape[0]):
            #     gt_1 = np.maximum(gt_1, make_gaussian((h, w), center=labels[ii, :], sigma=sigma))


            gt_0 = np.maximum(gt_0, make_gaussian((h, w), center=labels[0, :], sigma=sigma))

            # Euclidean distance map




            ### compute the Dahu distance map

            fg =  np.ones(shape=(h, w), dtype=np.uint8)
            fg[labels[1, :][1], :]  = 0
            fg[:, labels[1, :][0]]  = 0
            fg[labels[4, :][1], :]  = 0
            fg[:, labels[4, :][0]]  = 0
            dmap_fg = ndimage.distance_transform_edt(fg)

            dmap = np.exp(-4 * np.log(2) * (dmap_fg** 2) / sigma ** 2)
            gt_1 = np.maximum(gt_1, dmap)


            ### compute the Dahu distance map

            bg =  np.zeros(shape=(h, w), dtype=np.uint8)
            bg[labels[1, :][1], :]  = 255
            bg[:, labels[1, :][0]]  = 255
            bg[labels[4, :][1], :]  = 255
            bg[:, labels[4, :][0]]  = 255

            bg[0, :]  = 255
            bg[:, 0]  = 255
            bg[0, :]  = 255
            bg[:, 0]  = 255

            # height = h * 2 - 1
            # width = w * 2 - 1
            # F_bg = cv2.resize(bg, (width, height))
            # F_bg = np.array(F_bg, dtype="uint8")  # convert to uint8



            # dmap_scalar_bg = dahu.appro_dahu_scribble(image, F_bg)
            dmap_scalar_bg = MBD.MBD_waterflow(image, bg)

            dmap_scalar_bg[0:labels[1, :][1], :]  = 0
            dmap_scalar_bg[:, 0:labels[1, :][0]]  = 0
            dmap_scalar_bg[labels[4, :][1]:, :]  = 0
            dmap_scalar_bg[:, labels[4, :][0]:]  = 0

            dmap = dmap_scalar_bg/ (np.max(dmap_scalar_bg) + 0.001) * 255

            ############BIC

            (n_rows, n_cols, n_channels) = image.shape
            img_size = math.sqrt(n_rows * n_cols)
            border_thickness = int(math.floor(0.02 * img_size))

            img_lab = img_as_float(rgb2lab(image))
            # img_lab = img_as_float(img)


            # A = np.sum(img_lab, axis=2)
            # A[labels[1, :][1]:labels[4, :][1], labels[1, :][0]:labels[4, :][0]]  = 0
            #
            # nonzero = np.nonzero(A)
            # nonzero_row = nonzero[0]
            # nonzero_col = nonzero[1]
            # color = []
            # for row, col in zip(nonzero_row, nonzero_col):
            #     color.append(img_lab[row,col])
            #
            # u = np.zeros((h, w))
            # if color:
            #     h, w, nchannel = img_lab.shape
            #     n_comp = 5
            #
            #     gmm_bg = mixture.BayesianGaussianMixture(n_comp, covariance_type='full', tol=0.001, random_state=0)
            #     gmm_bg.fit(color)
            #     print("bg:", gmm_bg.weights_)
            #
            #     u = gmm_bg.score_samples(img_lab.reshape((w * h, nchannel)))
            #     u = u.reshape((h, w))







            #################### 4 borders

            px_left = img_lab[0:border_thickness, :, :]
            px_right = img_lab[n_rows - border_thickness - 1:-1, :, :]
            px_top = img_lab[:, 0:border_thickness, :]
            px_bottom = img_lab[:, n_cols - border_thickness - 1:-1, :]

            px_mean_left = np.mean(px_left, axis=(0, 1))
            px_mean_right = np.mean(px_right, axis=(0, 1))
            px_mean_top = np.mean(px_top, axis=(0, 1))
            px_mean_bottom = np.mean(px_bottom, axis=(0, 1))

            px_left = px_left.reshape((n_cols * border_thickness, 3))
            px_right = px_right.reshape((n_cols * border_thickness, 3))
            px_top = px_top.reshape((n_rows * border_thickness, 3))
            px_bottom = px_bottom.reshape((n_rows * border_thickness, 3))


            cov_left = np.cov(px_left.T)
            cov_right = np.cov(px_right.T)

            cov_top = np.cov(px_top.T)
            cov_bottom = np.cov(px_bottom.T)

            cov_left = np.linalg.pinv(cov_left)
            cov_right = np.linalg.pinv(cov_right)

            cov_top = np.linalg.pinv(cov_top)
            cov_bottom = np.linalg.pinv(cov_bottom)

            u_left = np.zeros((h,w))
            u_right = np.zeros((h,w))
            u_top = np.zeros((h,w))
            u_bottom = np.zeros((h,w))

            u_final = np.zeros((h,w))
            img_lab_unrolled = img_lab.reshape(img_lab.shape[0] * img_lab.shape[1], 3)

            if np.sum(px_mean_left) > 0:
                px_mean_left_2 = np.zeros((1, 3))
                px_mean_left_2[0, :] = px_mean_left
                u_left = scipy.spatial.distance.cdist(img_lab_unrolled, px_mean_left_2, 'mahalanobis', VI=cov_left)
                u_left = u_left.reshape((img_lab.shape[0], img_lab.shape[1]))
            else:
                u_left = np.zeros((h, w))

            if np.sum(px_mean_right) > 0:
                px_mean_right_2 = np.zeros((1, 3))
                px_mean_right_2[0, :] = px_mean_right
                u_right = scipy.spatial.distance.cdist(img_lab_unrolled, px_mean_right_2, 'mahalanobis', VI=cov_right)
                u_right = u_right.reshape((img_lab.shape[0], img_lab.shape[1]))
            else:
                u_right = np.zeros((h, w))

            if np.sum(px_mean_top) > 0:
                px_mean_top_2 = np.zeros((1, 3))
                px_mean_top_2[0, :] = px_mean_top
                u_top = scipy.spatial.distance.cdist(img_lab_unrolled, px_mean_top_2, 'mahalanobis', VI=cov_top)
                u_top = u_top.reshape((img_lab.shape[0], img_lab.shape[1]))
            else:
                u_top = np.zeros((h, w))

            if np.sum(px_mean_bottom) > 0:
                px_mean_bottom_2 = np.zeros((1, 3))
                px_mean_bottom_2[0, :] = px_mean_bottom
                u_bottom = scipy.spatial.distance.cdist(img_lab_unrolled, px_mean_bottom_2, 'mahalanobis', VI=cov_bottom)
                u_bottom = u_bottom.reshape((img_lab.shape[0], img_lab.shape[1]))
            else:
                u_bottom = np.zeros((h, w))

            max_u_left = np.max(u_left)
            max_u_right = np.max(u_right)
            max_u_top = np.max(u_top)
            max_u_bottom = np.max(u_bottom)

            u_left = u_left / (max_u_left + 0.001)
            u_right = u_right / (max_u_right + 0.001)
            u_top = u_top / (max_u_top + 0.001)
            u_bottom = u_bottom / (max_u_bottom + 0.001)

            u_max = np.maximum(np.maximum(np.maximum(u_left, u_right), u_top), u_bottom)
            u_min = np.minimum(np.minimum(np.minimum(u_left, u_right), u_top), u_bottom)

            u_final = (u_left + u_right + u_top + u_bottom) - u_max
            u_final = u_final/ (np.max(u_final) + 0.001) * 255
            u_final = np.asarray(u_final, dtype=np.uint8)
            u_final = cv2.medianBlur(u_final, 3)





            u_final[0:labels[1, :][1], :]  = 0
            u_final[:, 0:labels[1, :][0]]  = 0
            u_final[labels[4, :][1]:, :]  = 0
            u_final[:, labels[4, :][0]:]  = 0

            u_final = u_final/ (np.max(u_final) + 0.001) * 255


            ### sum

            sal = u_final/ (np.max(u_final) + 0.001) +    dmap/ (np.max(dmap) + 0.001)

            xv, yv = np.meshgrid(np.arange(dmap_scalar_bg.shape[1]), np.arange(dmap_scalar_bg.shape[0]))
            w2 = labels[0, 0]
            h2 = labels[0, 1]

            C = 1 - np.sqrt(np.power(xv - w2, 2) + np.power(yv - h2, 2)) / math.sqrt(np.power(w2, 2) + np.power(h2, 2))

            sal = sal * C

            sal = sal/ (np.max(sal) + 0.001) * 255


            gt_2 = np.maximum(gt_2, sal)


            
    gt = np.zeros(shape=(h, w, 3))
    gt[:, :, 0]=gt_0
    gt[:, :, 1]=gt_1
    gt[:, :, 2]=gt_2


    # gt = gt.astype(dtype=img.dtype) #(0~1)
    return gt  

def cstm_normalize(im, max_value):
    """
    Normalize image to range 0 - max_value
    """
    imn = max_value*(im - im.min()) / max((im.max() - im.min()), 1e-8)
    return imn


def generate_param_report(logfile, param):
    log_file = open(logfile, 'w')
    for key, val in param.items():
        log_file.write(key+':'+str(val)+'\n')
    log_file.close()
