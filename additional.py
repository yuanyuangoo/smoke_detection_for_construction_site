#%%
import requests
from numpy.lib.utils import source
from semantic import sem_mask
import uuid
import time
import cv2
import re
import os
import numpy as np
import torch


def split_with_mask(image, image_size, mask=None, split_points=None, for_reduce=False):

    results, out_split_points = [], []
    w, h = image.shape[3], image.shape[2]

    if mask is None:
        mask = ((0, 0), (w, h))
    xmax = mask[1][0]
    ymax = mask[1][1]
    xmin = mask[0][0]
    ymin = mask[0][1]
    if split_points == None:
        for i in range(int(w/image_size)+1):
            for j in range(int(h/image_size)+1):
                x1, x2, y1, y2 = i * \
                    image_size, (i+1)*image_size, j * \
                    image_size, (j+1)*image_size
                x1_clip = np.clip(x1, xmin, xmax)
                x2_clip = np.clip(x2, xmin, xmax)
                y1_clip = np.clip(y1, ymin, ymax)
                y2_clip = np.clip(y2, ymin, ymax)
                if x2_clip-x1_clip > 20 and y2_clip-y1_clip > 20:
                    if for_reduce:
                        if len(torch.unique(image[:, :, y1: y2, x1: x2])) == 0:
                            continue
                    results.append(image[:, :, y1: y2, x1: x2])
                    out_split_points.append((x1, y1))
    else:
        for item in split_points:
            x1, x2, y1, y2 = item[0], item[0]+512, item[1], item[1]+512
            x1_clip = np.clip(x1, xmin, xmax)
            x2_clip = np.clip(x2, xmin, xmax)
            y1_clip = np.clip(y1, ymin, ymax)
            y2_clip = np.clip(y2, ymin, ymax)

            if x2_clip-x1_clip > 20 and y2_clip-y1_clip > 20:
                if for_reduce:
                    if len(torch.unique(image[:, :, y1: y2, x1: x2])) == 0:
                        continue
                results.append(image[:, :, y1: y2, x1: x2])
                out_split_points.append((x1, y1))
    return torch.cat(results, axis=0), out_split_points


def reverse_coords(coords, img_size, split_point):

    tmp = torch.zeros_like(coords)
    tmp[:, 0] = split_point[0]
    tmp[:, 1] = split_point[1]
    tmp[:, 2] = split_point[0]
    tmp[:, 3] = split_point[1]
    return coords + tmp

# %%


def get_mask(refPt, imgsz=512, dialate=False):
    split_points = ((192, 0), (192, 284), (192, 568),
                    (704, 0), (704, 284), (704, 568),
                    (1216, 0), (1216, 284), (1216, 568))
    get_mask_ = True
    t0 = time.time()
    if get_mask_:
        while True:
            answer = input("get mask from file, y or n: ")
            if answer == "y":
                # Do this.
                get_mask_from_file = True
                break
            elif answer == "n":
                # Do that.
                get_mask_from_file = False
                break
            else:
                get_mask_from_file = True
                break
        if not get_mask_from_file or not os.path.exists('mask.jpg'):
            mask = sem_mask()
            cv2.imwrite('mask.jpg', mask*255)
            print('sem_mask Done. (%.3fs)' % (time.time() - t0))
        else:
            mask = cv2.imread('mask.jpg', 0)/255
        if dialate:
            kernel = np.ones((15, 15))
            mask = cv2.dilate(mask*255, kernel, iterations=7)/255
        _, split_points = split_with_mask(
            torch.from_numpy(np.expand_dims(np.expand_dims(mask, axis=0), axis=0)), imgsz, refPt, split_points, for_reduce=True)
    else:
        mask = None
    return mask, split_points


marking = False


def draw_rect(event, x, y, flags, param):
    # grab references to the global variables
    global refPt, marking, image, ori_img, scale
    # if the left mouse button was clicked, record the starting
    # (x, y) coordinates and indicate that cropping is being
    # performed
    if event == cv2.EVENT_LBUTTONDOWN:
        if marking:
            image = ori_img.copy()
        refPt = [(int(x*ori_img.shape[1]/scale[0]),
                  int(y*ori_img.shape[0]/scale[1]))]

    # check to see if the left mouse button was released
    elif event == cv2.EVENT_LBUTTONUP:
        # record the ending (x, y) coordinates and indicate that
        # the cropping operation is finished
        refPt.append((int(x*ori_img.shape[1]/scale[0]),
                      int(y*ori_img.shape[0]/scale[1])))
        marking = True
        # draw a rectangle around the region of interest
        cv2.rectangle(image, refPt[0], refPt[1], (0, 255, 0),  thickness=1)
        image_reszie = cv2.resize(image, scale)
        cv2.imshow('image', image_reszie)


def get_bounding_box(source, scale_):
    global refPt, marking, image, ori_img, scale

    while True:
        answer = input("get_bounding_from_file, y or n: ")
        if answer == "y":
            # Do this.
            get_bounding_from_file = True
            break
        elif answer == "n":
            # Do that.
            get_bounding_from_file = False
            break
        else:
            get_bounding_from_file = True
            break
    if source.endswith('.txt') or source.endswith('.ip'):
        with open(source, 'r') as f:
            source = [x.strip()
                        for x in f.read().splitlines() if len(x.strip())]
    cap = cv2.VideoCapture(source[0])
    print(source[0])
    assert cap.isOpened(), 'Failed to open %s'.format(source[0])
    get_first_frame, first_frame = cap.read()  # guarantee first frame
    assert get_first_frame and first_frame is not None, 'Failed to get first frame'
    ori_img = first_frame.copy()
    images_size = ori_img.shape[:2]
    refPt='Not defined'

    # get_bounding_from_file = True
    if not os.path.exists('bounding_box.txt') or os.stat("bounding_box.txt").st_size == 0:
        get_bounding_from_file = False
    if get_bounding_from_file:
        with open("bounding_box.txt", "r") as fo:
            tmp = fo.read()
            tmp = re.findall(r'\d+', tmp)
            refPt = ((int(tmp[0]), int(tmp[1])), (int(tmp[2]), int(tmp[3])))
    else:

        image = first_frame
        scale = scale_
        cv2.namedWindow('image')
        cv2.setMouseCallback('image', draw_rect)

        while (1):
            image_reszie = cv2.resize(image, scale)
            cv2.imshow('image', image_reszie)

            k = cv2.waitKey(20) & 0xFF
            if k == 27:
                cv2.destroyWindow('image')
                break
            elif k == ord('a'):
                print(refPt)

        with open("bounding_box.txt", "w") as fo:
            for item in refPt:
                fo.write("%s\n" % str(item))

    refPt = ((max(min(refPt[0][0], refPt[1][0]), 0),
              max(min(refPt[0][1], refPt[1][1]), 0)), (min(max(refPt[0][0], refPt[1][0]), images_size[1]),
                                                       min(max(refPt[0][1], refPt[1][1]), images_size[0])))
    return refPt

# %%
