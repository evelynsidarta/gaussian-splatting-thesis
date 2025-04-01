import os
import argparse
import numpy as np
import pathlib
import torch.nn as nn
import OpenEXR as exr
import Imath
from PIL import Image
import cv2
# from torch.utils.tensorboard import SummaryWriter

NAMING_TAIL = "0022"
VERBOSE = True

def read_depth_exr_file(filepath: pathlib.Path):
    
    exrfile = exr.InputFile(filepath.as_posix())
    raw_bytes = exrfile.channel('B', Imath.PixelType(Imath.PixelType.FLOAT))
    depth_vector = np.frombuffer(raw_bytes, dtype=np.float32)
    height = exrfile.header()['displayWindow'].max.y + 1 - exrfile.header()['displayWindow'].min.y
    width = exrfile.header()['displayWindow'].max.x + 1 - exrfile.header()['displayWindow'].min.x
    depth_map = np.reshape(depth_vector, (height, width))

    # normalized
    # depth_min = depth_map.min()
    # depth_max = depth_map.max()
    # max_val = (2**(8*2)) - 1
    
    # if depth_max - depth_min > np.finfo("float").eps:
    #     depth_map = max_val * (depth_map - depth_min) / (depth_max - depth_min)
    # else:
    #     depth_map = np.zeros(depth_map.shape, dtype=depth_map.dtype)

    # depth_map = -1 * (depth_map - np.min(depth_map)) / (np.max(depth_map) - np.min(depth_map))
    # depth_map = depth_map + 1

    # print(depth_map)

    return depth_map

def readImages(img_dir, gt_dir, extension_img = ".png", extension_gt = ".exr"):
    images = []
    gts = []

    # populate gt list, gt provided is in metric distance
    for idx, _ in enumerate(os.listdir(gt_dir)):
        gt_name = os.path.join(gt_dir, "depth" + str(idx) + NAMING_TAIL + extension_gt)
        gt_retrieved = read_depth_exr_file(pathlib.Path(gt_name))
        gts.append(gt_retrieved.copy())

    # open the img_dir and gt_dir files and compile the files into np.array inside the files into two separate arrays
    for idx, _ in enumerate(os.listdir(img_dir)):
        img_name = os.path.join(img_dir, "image" + str(idx) + NAMING_TAIL + extension_img)
        print("\nretrieving image from " + img_name)
        images.append(cv2.imread(img_name)[:, :, 0].copy())
    print("length: " + str(len(images)) + ", " + str(len(gts)))
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

    if VERBOSE:
        print(gt)

    return dict(a1=a1, a2=a2, a3=a3, abs_rel=abs_rel, rmse=rmse, log_10=log_10, rmse_log=rmse_log,
                silog=silog, sq_rel=sq_rel)

# transform shape of gt and pred, then input to calculate metrics

def evaluate_single(gt, pred, interpolate=True, min_depth_eval=0.1, max_depth_eval=10, **kwargs):
    if 'config' in kwargs:
        config = kwargs['config']
        min_depth_eval = config.min_depth_eval
        max_depth_eval = config.max_depth_eval
    
    if VERBOSE:
        print("\ngt shape: " + str(gt.shape) + ", pred shape: " + str(pred.shape))

    if gt.shape[-2:] != pred.shape[-2:] and interpolate:
        pred = nn.functional.interpolate(
            pred, gt.shape[-2:], mode='bilinear', align_corners=True)
        
    pred = pred.squeeze()
    pred[pred < min_depth_eval] = min_depth_eval
    pred[pred > max_depth_eval] = max_depth_eval
    pred[np.isinf(pred)] = max_depth_eval
    pred[np.isnan(pred)] = min_depth_eval

    print("\npred squeezed: ")
    print(pred)

    gt[gt < min_depth_eval] = min_depth_eval
    gt[gt > max_depth_eval] = max_depth_eval
    gt[np.isinf(gt)] = max_depth_eval
    gt[np.isnan(gt)] = min_depth_eval

    print("\gt squeezed: ")
    print(gt)

    return evaluate_metrics(gt, pred)

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("-gt", "--ground_truth", type=str,
                        required=True, help="Name of the directory containing gt images")
    parser.add_argument("-p", "--prediction", type=str,
                        required=True, help="training results directory.")
    args, _ = parser.parse_known_args()
    
    pred, gt = readImages(args.prediction, args.ground_truth)
    compiled = list(zip(gt, pred))

    # create dict to store results
    results = {'a1': np.array([0.0], dtype=np.float64), 'a2': np.array([0.0], dtype=np.float64), 'a3' : np.array([0.0], dtype=np.float64),
               'abs_rel' : np.array([0.0], dtype=np.float64), 'rmse' : np.array([0.0], dtype=np.float64), 'log_10' : np.array([0.0], dtype=np.float64),
               'sq_rel' : np.array([0.0], dtype=np.float64), 'rmse_log' : np.array([0.0], dtype=np.float64), 'silog' : np.array([0.0], dtype=np.float64)}
    n_samples = 0

    for idx, _ in enumerate(compiled):

        gt_single, pred_single = compiled[idx]
        val_metrics = evaluate_single(gt_single, pred_single)
        # print out results
        if VERBOSE:
            print("\nResults for image " + str(idx) + ":\n")
            for k,v in sorted(val_metrics.items(), key=lambda x: x[1], reverse=True):
                print(k,v)

        for k in val_metrics.keys():
            results[k] += val_metrics[k]
        n_samples += 1

    # average out the results
    for k in results.keys():
        results[k] = results[k] / n_samples

    # TODO: print out the results
    print("\nAverage results for this dataset:\n")
    for k,v in sorted(results.items(), key=lambda x: x[1], reverse=True):
        print(k,v)