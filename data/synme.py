from .config import HOME
import os.path as osp
import sys
import torch
import torch.utils.data as data
import cv2
import numpy as np
if sys.version_info[0] == 2:
    import xml.etree.cElementTree as ET
else:
    import xml.etree.ElementTree as ET

SYNME_CLASSES = tuple(range(211))

SYNME_ROOT = osp.join(HOME, 'data/synme/')

class synmeAnnotationTransform(object):

    def __init__(self, class_to_ind=None):
        self.class_to_ind = class_to_ind or dict(
            zip(SYNME_CLASSES, range(len(SYNME_CLASSES))))

    def __call__(self, target, width, height):
        """
        Arguments:
            target (annotation) : the target annotation to be made usable
                will be an ET.Element
        Returns:
            a list containing lists of bounding boxes  [bbox coords, class name]
        """
        res = []
        for obj in target.iter('object'):
            name = obj.find('name').text.lower().strip()
            name = int(name)
            bbox = obj.find('bndbox')

            pts = ['xmin', 'ymin', 'xmax', 'ymax']
            bndbox = []
            for i, pt in enumerate(pts):
                cur_pt = int(bbox.find(pt).text) - 1
                # scale height or width
                cur_pt = cur_pt / width if i % 2 == 0 else cur_pt / height
                bndbox.append(cur_pt)
            label_idx = self.class_to_ind[name]
            bndbox.append(label_idx)
            res += [bndbox]  # [xmin, ymin, xmax, ymax, label_ind]

        return res  # [[xmin, ymin, xmax, ymax, label_ind], ... ]


class synmeDetection(data.Dataset):
    """

    input is image, target is annotation

    Arguments:
        root (string): filepath to VOCdevkit folder.
        image_set (string): imageset to use (eg. 'train', 'val', 'test')
        target_transform (callable, optional): transformation to perform on the
            target `annotation`
            (eg: take in caption string, return tensor of word indices)
        dataset_name (string, optional): which dataset to load
            (default: 'VOC2007')
    """

    def __init__(self, root='/lustre/home/lpwang/data/synme', transform=None, target_transform=synmeAnnotationTransform(),
                 dataset_name='synme', instance_file='train.txt'):
        self.root = root
        self.transform = transform
        self.target_transform = target_transform
        self.name = dataset_name
        self.pairs = list()
        for line in open(osp.join(root, instance_file)):
            line = line.strip().split()
            self.pairs.append(tuple(line))

    def __getitem__(self, index):
        im, gt, h, w = self.pull_item(index)

        return im, gt

    def __len__(self):
        return len(self.pairs)

    def pull_item(self, index):
        pair = self.pairs[index]
        img_path = osp.join(self.root, pair[0])
        target_path = osp.join(self.root, pair[1])
        target = ET.parse(target_path).getroot()
        img = cv2.imread(img_path) # H W C(BGR)
        height, width, channels = img.shape

        if self.target_transform is not None:
            target = self.target_transform(target, width, height)

        if self.transform is not None:
            target = np.array(target)
            img, boxes, labels = self.transform(img, target[:, :4], target[:, 4])
            target = np.hstack((boxes, np.expand_dims(labels, axis=1)))

        img = img[:, :, (2,1,0)] # BGR to RGB
        return torch.from_numpy(img).permute(2, 0, 1), target, height, width # C H W
