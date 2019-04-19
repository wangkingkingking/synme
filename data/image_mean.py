
import numpy as np
import cv2
import os
import random
import argparse

parser = argparse.ArgumentParser(description = 'Calculate an RGB mean estimation of all images in a dir')
parser.add_argument('--image_dir')
parser.add_argument('-r','--ratio', default = 0.1,  type=float, help='ratio of calculated image num to the total num of images')
parser.add_argument('--extention', type=str, default='.jpg')
parser.add_argument('-m', '--max', default = None)
args = parser.parse_args()


def main():
    image_dir = args.image_dir
    imglist = os.listdir(image_dir)
    num = int(len(imglist)*args.ratio)
    random.shuffle(imglist)

    res = [0, 0, 0]
    cnt = 0
    for img in imglist:
        if not img.endswith(args.extention):
            continue
        cnt += 1
        if cnt == num:
            break
        img = os.path.join(image_dir, img)
        img = cv2.imread(img)
        H, W, C = img.shape
        img = img.sum(0) / H
        img = img.sum(0) / W
        res = res + img/num
    print('BGR mean is ', res)
    return res


if __name__ == '__main__':
    main()

