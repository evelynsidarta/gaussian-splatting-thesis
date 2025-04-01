import os
import argparse
import numpy as np
import pathlib
import torch.nn as nn
import OpenEXR as exr
import Imath
import matplotlib.pyplot as plt
from PIL import Image
import cv2
# from torch.utils.tensorboard import SummaryWriter

NAMING_TAIL1 = "0022"
NAMING_TAIL2 = "0022"
VERBOSE = True

os.environ["OPENCV_IO_ENABLE_OPENEXR"]="1"

def inverse_depth(x, epsilon = 1e-3):
    return 1 / (x + epsilon)

def read_depth_exr_file(filepath: pathlib.Path):
    
    exrfile = exr.InputFile(filepath.as_posix())
    raw_bytes = exrfile.channel('B', Imath.PixelType(Imath.PixelType.FLOAT))
    depth_vector = np.frombuffer(raw_bytes, dtype=np.float32)
    height = exrfile.header()['displayWindow'].max.y + 1 - exrfile.header()['displayWindow'].min.y
    width = exrfile.header()['displayWindow'].max.x + 1 - exrfile.header()['displayWindow'].min.x
    depth_map = np.reshape(depth_vector, (height, width))

    return depth_map

def read_depth_tiff_file(filepath: pathlib.Path):
    pass

def readImages(img_dir, gt_dir, extension_img = ".png", extension_gt = ".exr"):
    images = []
    gts = []

    # populate gt list, gt provided is in metric distance
    for idx, _ in enumerate(os.listdir(gt_dir)):
        gt_name = os.path.join(gt_dir, "depth" + str(idx) + NAMING_TAIL1 + extension_gt)
        #gt_retrieved = read_depth_exr_file(pathlib.Path(gt_name))
        gt_retrieved = cv2.imread(gt_name)[:, :, 0].copy()
        # gt_retrieved = np.where(gt_retrieved >= np.max(gt_retrieved), (0.5 * gt_retrieved), gt_retrieved)
        gt_retrieved = inverse_depth(gt_retrieved)
        gt_retrieved = (gt_retrieved - np.min(gt_retrieved)) / (np.max(gt_retrieved) - np.min(gt_retrieved))
        gts.append(gt_retrieved.copy())
        # gt_data = cv2.imread(gt_name)[:, :, 0].copy()
        # plt.imshow(gt_data)
        # plt.show()
        # gts.append(gt_data)

    # open the img_dir and gt_dir files and compile the files into np.array inside the files into two separate arrays
    for idx, _ in enumerate(os.listdir(img_dir)):
        img_name = os.path.join(img_dir, "image" + str(idx) + NAMING_TAIL2 + extension_img)
        # print("\nretrieving image from " + img_name)
        img_data = cv2.resize(cv2.imread(img_name)[:, :, 0].copy(), (1920, 1080))
        img_data = (img_data - np.min(img_data)) / (np.max(img_data) - np.min(img_data))
        # plt.imshow(img_data)
        # plt.show()
        images.append(img_data)
        
    # print("length: " + str(len(images)) + ", " + str(len(gts)))
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

def evaluate_single(gt, pred, interpolate=True, min_depth_eval=0.1, max_depth_eval=100, **kwargs):
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

    gt = gt.squeeze()
    gt[gt < min_depth_eval] = min_depth_eval
    gt[gt > max_depth_eval] = max_depth_eval
    gt[np.isinf(gt)] = max_depth_eval
    gt[np.isnan(gt)] = min_depth_eval

    print("\gt squeezed: ")
    print(gt)

    #gt_depth = gt.squeeze()
    #valid_mask = np.logical_and(
    #    gt_depth > min_depth_eval, gt_depth < max_depth_eval)
    #eval_mask = np.ones(valid_mask.shape)
    #valid_mask = np.logical_and(valid_mask, eval_mask)

    #print("\ngt_depth valid_mask:")
    #print(gt_depth[valid_mask])
    #return evaluate_metrics(gt_depth[valid_mask], pred[valid_mask])
    return evaluate_metrics(gt, pred)

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("-gt", "--ground_truth", type=str,
                        required=True, help="Name of the directory containing gt images")
    parser.add_argument("-p", "--prediction", type=str,
                        required=True, help="training results directory.")
    #parser.add_argument("-o", "--output", type=str,
    #                    required=True, help="Directory for metrics output")
    args, _ = parser.parse_known_args()
    
    pred, gt = readImages(args.prediction, args.ground_truth)
    #plt.imshow(pred[0])
    #plt.show()
    #plt.imshow(gt[0])
    #plt.show()
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