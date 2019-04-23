import torch
import sys
import os
import time
import argparse
import numpy as np
import pickle
import cv2

from torch import nn
from torch.backends import cudnn
from torch.autograd import Variable

from config import DATASET_ROOT, MEANS, EVAL_DIR, synme
from data import  AnnotationTransform, Dataset, ZeroMeanTransform
from data import SYNME_CLASSES as classnames
from ssd import build_ssd
from utils import str2bool, output_file, load_weights, Timer


parser = argparse.ArgumentParser(
    description='Single Shot MultiBox Detector Evaluation')
parser.add_argument('--trained_model', type=str,
                    help='Trained state_dict file path to open')
parser.add_argument('--save_folder', default=EVAL_DIR, type=str,
                    help='File path to save results')
parser.add_argument('--confidence_threshold', default=0.01, type=float,
                    help='Detection confidence threshold')
parser.add_argument('--top_k', default=5, type=int,
                    help='Further restrict the number of predictions to parse')
parser.add_argument('--cuda', default=True, type=str2bool,
                    help='Use cuda to train model')
parser.add_argument('--cleanup', default=True, type=str2bool,
                    help='Cleanup and remove results files following eval')
args = parser.parse_args()


if not os.path.isdir(args.save_folder):
    os.mkdir(args.save_folder)

if torch.cuda.is_available():
    if args.cuda:
        torch.set_default_tensor_type('torch.cuda.FloatTensor')
    if not args.cuda:
        print("WARNING: It looks like you have a CUDA device, but aren't using \
              CUDA.  Run with --cuda for optimal eval speed.")
        torch.set_default_tensor_type('torch.FloatTensor')
else:
    torch.set_default_tensor_type('torch.FloatTensor')

phase = 'test'
det_pkl_file = os.path.join(args.save_folder, 'detections.pkl')

def class_dec_result_filepath(cls):
    filename = 'det_' + 'class_%s.txt' % (cls)
    return  os.path.join(args.save_dir, filename)


def write_dec_results(all_boxes, dataset):
    for cls_ind, cls_name in enumerate(classnames):
        if cls_ind == 0: #background
            continue
        print('Writing class {:s} synme results file'.format(cls))
        filename = class_dec_result_file(args.save_dir, cls_name)
        with open(filename, 'w') as f:
            for im_ind, pair in enumerate(dataset.pairs):
                dets = all_boxes[cls_ind][im_ind]
                if dets == []:
                    continue
                # the original annotation start index from 1
                for k in range(dets.shape[0]):
                    f.write('{:s} {:.3f} {:.1f} {:.1f} {:.1f} {:.1f}\n'.
                            format(pair[0].split('/')[1], dets[k, -1],
                                   dets[k, 0] + 1, dets[k, 1] + 1,
                                   dets[k, 2] + 1, dets[k, 3] + 1))


def eval_value():
    test_list_file = os.path.join(DATASET_ROOT, 'test.txt') 
    aps = []
    for i, cls in enumerate(classnames):
        if i==0:
            continue #background
        filename = class_dec_result_file(cls)
        rec, prec, ap = eval_cls(
           filename, annopath, test_list_file, cls, cachedir,
           ovthresh=0.5, use_07_metric=use_07_metric)
        aps += [ap]
        print('AP for {} = {:.4f}'.format(cls, ap))
        with open(os.path.join(eval_value_dir, cls + '_pr.pkl'), 'wb') as f:
            pickle.dump({'rec': rec, 'prec': prec, 'ap': ap}, f)
    print('Mean AP = {:.4f}'.format(np.mean(aps)))
    print('~~~~~~~~')
    print('Results:')
    for ap in aps:
        print('{:.3f}'.format(ap))
    print('{:.3f}'.format(np.mean(aps)))
    print('~~~~~~~~')
    print('')


def eval_cls(classname, ovthresh=0.5):
    # first load gt
    if not os.path.isdir(cachedir):
        os.mkdir(cachedir)

    anno_dic = get_anno_dic(cache_file, os.path.join(DATASET_ROOT, 'test.txt'))
    # extract gt objects for this class
    class_recs = {}
    npos = 0

    imagenames = get_image_name_list(test_list_file)
    for imagename in imagenames:
        R = [obj for obj in anno_dic[imagename] if obj['name'] == classname]
        bbox = np.array([x['bbox'] for x in R])
        det = [False] * len(R)
        class_recs[imagename] = {'bbox': bbox,
                                 'det': det}

    # read dets
    detfile = get_synme_results_file(phase, classname) 
    with open(detfile, 'r') as f:
        lines = f.readlines()
    if any(lines) == 1:
        splitlines = [x.strip().split(' ') for x in lines]
        image_ids = [x[0] for x in splitlines]
        confidence = np.array([float(x[1]) for x in splitlines])
        BB = np.array([[float(z) for z in x[2:]] for x in splitlines])

        # sort by confidence
        sorted_ind = np.argsort(-confidence)
        sorted_scores = np.sort(-confidence)
        BB = BB[sorted_ind, :]
        image_ids = [image_ids[x] for x in sorted_ind]

        # go down dets and mark TPs and FPs
        nd = len(image_ids)
        tp = np.zeros(nd)
        fp = np.zeros(nd)
        for d in range(nd):
            R = class_recs[image_ids[d]]
            bb = BB[d, :].astype(float)
            ovmax = -np.inf
            BBGT = R['bbox'].astype(float)
            if BBGT.size > 0:
                # compute overlaps
                # intersection
                ixmin = np.maximum(BBGT[:, 0], bb[0])
                iymin = np.maximum(BBGT[:, 1], bb[1])
                ixmax = np.minimum(BBGT[:, 2], bb[2])
                iymax = np.minimum(BBGT[:, 3], bb[3])
                iw = np.maximum(ixmax - ixmin, 0.)
                ih = np.maximum(iymax - iymin, 0.)
                inters = iw * ih
                uni = ((bb[2] - bb[0]) * (bb[3] - bb[1]) +
                       (BBGT[:, 2] - BBGT[:, 0]) *
                       (BBGT[:, 3] - BBGT[:, 1]) - inters)
                overlaps = inters / uni
                ovmax = np.max(overlaps)
                jmax = np.argmax(overlaps)

            if ovmax > ovthresh:
                if not R['difficult'][jmax]:
                    if not R['det'][jmax]:
                        tp[d] = 1.
                        R['det'][jmax] = 1
                    else:
                        fp[d] = 1.
            else:
                fp[d] = 1.

        # compute precision recall
        fp = np.cumsum(fp)
        tp = np.cumsum(tp)
        rec = tp / float(npos)
        # avoid divide by zero in case the first detection matches a difficult
        # ground truth
        prec = tp / np.maximum(tp + fp, np.finfo(np.float64).eps)
        ap = synme_ap(rec, prec, use_07_metric)
    else:
        rec = -1.
        prec = -1.
        ap = -1.

    return rec, prec, ap


def test_net(net, dataset, transform, top_k, thresh=0.05):

    num_images = len(dataset)
    # all detections are collected into:
    #    all_boxes[cls][image] = N x 5 array of detections in
    #    (x1, y1, x2, y2, score)
    all_boxes = [[[] for _ in range(num_images)]
                 for _ in range(len(classnames))]

    _t = {'im_detect': Timer(), 'misc': Timer()}

    for i in range(num_images):
        im, gt, h, w = dataset.pull_item(i)
        x = Variable(im.unsqueeze(0))
        _t['im_detect'].tic()
        detections = net(x).detach()
        detect_time = _t['im_detect'].toc(average=False)

        # skip j = 0, because it's the background class
        for j in range(1, detections.size(1)):
            dets = detections[0, j, :]
            mask = dets[:, 0].gt(0.).expand(5, dets.size(0)).t()
            dets = torch.masked_select(dets, mask).view(-1, 5)
            if dets.size(0) == 0:
                continue
            boxes = dets[:, 1:]
            boxes[:, 0] *= w
            boxes[:, 2] *= w
            boxes[:, 1] *= h
            boxes[:, 3] *= h
            scores = dets[:, 0].cpu().numpy()
            cls_dets = np.hstack((boxes.cpu().numpy(),
                                  scores[:, np.newaxis])).astype(np.float32,
                                                                 copy=False)
            all_boxes[j][i] = cls_dets

        print('im_detect: {:d}/{:d} {:.3f}s'.format(i + 1,
                                                    num_images, detect_time))

    det_file = os.path.join(output_dir, 'detections.pkl')
    with open(det_file, 'wb') as f:
        pickle.dump(all_boxes, f, pickle.HIGHEST_PROTOCOL)

    print('Evaluating detections...')
    write_synme_results_file()
    eval_value()



if __name__ == '__main__':

    net = build_ssd('test', synme)
    load_weights(net, args.trained_model)
    if args.cuda:
        net = net.cuda()
        cudnn.benchmark = True

    net.eval()
    print('Finished loading model!')

    dataset = Dataset(transform=ZeroMeanTransform(300, MEANS), instance_file = 'test.txt')

    if args.cleanup:
        raise NotImplementedError

    test_net(net, dataset,
             ZeroMeanTransform(net.size, MEANS), args.top_k, 
             thresh=args.confidence_threshold)
