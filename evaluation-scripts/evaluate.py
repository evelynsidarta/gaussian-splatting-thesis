import os
import argparse
import numpy as np
import pathlib
import torch.nn as nn
from utils.misc import read_depth_exr_file
from PIL import Image
from torch.utils.tensorboard import SummaryWriter

NAMING_TAIL = "0022"

def readImages(img_dir, gt_dir, extension_img = ".png", extension_gt = ".exr"):
    images = []
    gts = []

    # populate gt list, gt provided is in metric distance
    for idx, _ in enumerate(os.listdir(gt_dir)):
        gt_name = os.path.join(gt_dir, "depth" + str(idx) + NAMING_TAIL + extension_gt)
        gts.append(read_depth_exr_file(gt_name))

    # open the img_dir and gt_dir files and compile the files into np.array inside the files into two separate arrays
    for idx, _ in enumerate(os.listdir(img_dir)):
        img_name = os.path.join(img_dir, "image" + str(idx) + NAMING_TAIL + extension_img)
        images.append(np.array(Image.open(img_name)))

    return images, gts

""" Goal: return dictionary containing following metrics:
            'a1': Delta1 accuracy: Fraction of pixels that are within a scale factor of 1.25
            'a2': Delta2 accuracy: Fraction of pixels that are within a scale factor of 1.25^2
            'a3': Delta3 accuracy: Fraction of pixels that are within a scale factor of 1.25^3
            'abs_rel': Absolute relative error
            'rmse': Root mean squared error
            'log_10': Absolute log10 error
            'sq_rel': Squared relative error
            'rmse_log': Root mean squared error on the log scale
            'silog': Scale invariant log error
    Inputs: gt (np.ndarray) filled with gt values
            pred (np.ndarray)
            gt.shape = pred.shape
    Adapted from evaluate.py of ZoeDepth
"""

def evaluate_metrics(gt, pred):
    thresh = np.maximum((gt / pred), (pred / gt))
    a1 = (thresh < 1.25).mean()
    a2 = (thresh < 1.25 ** 2).mean()
    a3 = (thresh < 1.25 ** 3).mean()

    abs_rel = np.mean(np.abs(gt - pred) / gt)
    sq_rel = np.mean(((gt - pred) ** 2) / gt)

    rmse = (gt - pred) ** 2
    rmse = np.sqrt(rmse.mean())

    rmse_log = (np.log(gt) - np.log(pred)) ** 2
    rmse_log = np.sqrt(rmse_log.mean())

    err = np.log(pred) - np.log(gt)
    silog = np.sqrt(np.mean(err ** 2) - np.mean(err) ** 2) * 100

    log_10 = (np.abs(np.log10(gt) - np.log10(pred))).mean()
    return dict(a1=a1, a2=a2, a3=a3, abs_rel=abs_rel, rmse=rmse, log_10=log_10, rmse_log=rmse_log,
                silog=silog, sq_rel=sq_rel)

# transform shape of gt and pred, then input to calculate metrics

def evaluate_single(gt, pred, interpolate=True, min_depth_eval=0.1, max_depth_eval=10, **kwargs):
    
    if 'config' in kwargs:
        config = kwargs['config']
        min_depth_eval = config.min_depth_eval
        max_depth_eval = config.max_depth_eval
    
    if gt.shape[-2:] != pred.shape[-2:] and interpolate:
        pred = nn.functional.interpolate(
            pred, gt.shape[-2:], mode='bilinear', align_corners=True)
        
    pred = pred.squeeze().cpu().numpy()
    pred[pred < min_depth_eval] = min_depth_eval
    pred[pred > max_depth_eval] = max_depth_eval
    pred[np.isinf(pred)] = max_depth_eval
    pred[np.isnan(pred)] = min_depth_eval

    gt_depth = gt.squeeze().cpu().numpy()
    valid_mask = np.logical_and(
        gt_depth > min_depth_eval, gt_depth < max_depth_eval)
    eval_mask = np.ones(valid_mask.shape)
    valid_mask = np.logical_and(valid_mask, eval_mask)
    return evaluate_metrics(gt_depth[valid_mask], pred[valid_mask])

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("-gt", "--ground_truth", type=str,
                        required=True, help="Name of the directory containing gt images")
    parser.add_argument("-p", "--prediction", type=str,
                        required=True, help="training results directory.")
    parser.add_argument("-o", "--output", type=str,
                        required=True, help="Directory for metrics output")
    args, _ = parser.parse_known_args()
    
    pred, gt = readImages(args.prediction, args.ground_truth)
    compiled = list(zip(gt, pred))

    # create dict to store results
    results = {'a1': np.array([0.0], dtype=np.uint16), 'a2': np.array([0.0], dtype=np.uint16), 'a3' : np.array([0.0], dtype=np.uint16),
               'abs_rel' : np.array([0.0], dtype=np.uint16), 'rmse' : np.array([0.0], dtype=np.uint16), 'log_10' : np.array([0.0], dtype=np.uint16),
               'sq_rel' : np.array([0.0], dtype=np.uint16), 'rmse_log' : np.array([0.0], dtype=np.uint16), 'silog' : np.array([0.0], dtype=np.uint16)}
    n_samples = 0

    for idx, _ in enumerate(compiled):

        gt_single, pred_single = compiled[idx]
        val_metrics = evaluate_single(gt_single, pred_single)

        for k in val_metrics.keys():
            results[k] += val_metrics[k]
        n_samples += 1

    # average out the results


        # TODO: print out the results