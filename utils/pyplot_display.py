import cv2
import os
import argparse
import numpy as np
import matplotlib.pyplot as plt
os.environ["OPENCV_IO_ENABLE_OPENEXR"]="1"

# 1 = ground truth
# 2 = 3DGS
# 3 = 2DGS
# 4 = Depth Anything v1 Base
# 5 = Depth Anything v1 Large
# 6 = Depth Anything v1 Small
# 7 = Depth Anything v2 Base
# 8 = Depth Anything v2 Large
# 9 = Depth Anything v2 Small
# 10 = Zoe Depth

### upcoming TODOs:
# 11 = RaDe-GS
# 12 = gs2mesh

FORMAT_1 = ".exr"
FORMAT_2 = ".tiff"
FORMAT_3 = ".tiff"
FORMAT_4 = ".png"
FORMAT_5 = ".png"
FORMAT_6 = ".png"
FORMAT_7 = ".png"
FORMAT_8 = ".png"
FORMAT_9 = ".png"
FORMAT_10 = ".png"
# FORMAT_11 = ".tiff"
# FORMAT_12 = ".tiff"
# FORMAT_13 = ".tiff"
# FORMAT_14 = ".tiff"

def sigmoid(x, k = 10, x0 = 0.5):
    return 1 / (1 + np.exp(-k * (x - x0)))

def log_transform(x, alpha = 10):
    return 1 / (1 + alpha * x) / np.log(1 + alpha)

def inverse_depth(x, epsilon = 1e-3):
    return 1 / (x + epsilon)

def piecewise_linear(x, x_min = 0.2, x_max = 0.8):
    return np.clip((x - x_min) / (x_max - x_min), 0, 1)

def read_images_data(img_dir):
    
    # 1 = ground truth
    img_name = os.path.join(img_dir, "gt" + FORMAT_1)
    img_d1 = cv2.imread(img_name)[:, :, 0].copy()
    # transformations for gt:
    #   1. hard clamp the max values so it is not too exaggerated
    #   2. invert the depth map
    #   3. normalize so that the min value of the map is 0
    img_d1 = np.where(img_d1 >= np.max(img_d1), (0.5 * img_d1), img_d1)
    img_d1 = inverse_depth(img_d1)
    img_d1 = (img_d1 - np.min(img_d1)) / (np.max(img_d1) - np.min(img_d1))
    # display
    plt.subplot(4, 3, 1)
    plt.imshow(img_d1)
    plt.axis('off')
    plt.title("Ground Truth", fontsize=12)

    # 2 = 3DGS
    # img_name = os.path.join(img_dir, "3dgs" + FORMAT_2)
    # img_d2 = cv2.imread(img_name)[:, :, 0].copy()
    # plt.subplot(4, 3, 2)
    # plt.imshow(img_d2)
    # plt.axis('off')
    # plt.title("3DGS", fontsize=12)

    # 3 = 2DGS
    img_name = os.path.join(img_dir, "2dgs" + FORMAT_3)
    img_d3 = cv2.resize(cv2.imread(img_name, flags=(cv2.IMREAD_GRAYSCALE | cv2.IMREAD_ANYDEPTH)).copy(), 
                            (1920, 1080), interpolation = cv2.INTER_NEAREST)
    # transformations for 2dgs:
    #   1. Normalize values to [0,1]
    #   2. In the format saved by 2DGS, infinite value is written as 0 -> fix the value to max value 1
    #   3. invert the depth map
    #   4. normalize so that the min value of the map is 0
    img_d3 = img_d3 / np.max(img_d3)
    img_d3 = np.where(img_d3 <= np.min(img_d3), 1., img_d3)
    img_d3 = inverse_depth(img_d3)
    img_d3 = (img_d3 - np.min(img_d3)) / (np.max(img_d3) - np.min(img_d3))
    # display
    plt.subplot(4, 3, 3)
    plt.imshow(img_d3)
    plt.axis('off')
    plt.title("2DGS", fontsize=12)

    # 4 = Depth Anything v1 Base
    img_name = os.path.join(img_dir, "DA1b" + FORMAT_4)
    img_d4 = cv2.imread(img_name)[:, :, 0].copy()
    plt.subplot(4, 3, 4)
    plt.imshow(img_d4)
    plt.axis('off')
    plt.title("Depth Anything v1 Base", fontsize=12)

    # 5 = Depth Anything v1 Large
    img_name = os.path.join(img_dir, "DA1l" + FORMAT_5)
    img_d5 = cv2.imread(img_name)[:, :, 0].copy()
    plt.subplot(4, 3, 5)
    plt.imshow(img_d5)
    plt.axis('off')
    plt.title("Depth Anything v1 Large", fontsize=12)

    # 6 = Depth Anything v1 Small
    img_name = os.path.join(img_dir, "DA1s" + FORMAT_6)
    img_d6 = cv2.imread(img_name)[:, :, 0].copy()
    plt.subplot(4, 3, 6)
    plt.imshow(img_d6)
    plt.axis('off')
    plt.title("Depth Anything v1 Small", fontsize=12)

    # 7 = Depth Anything v2 Base
    img_name = os.path.join(img_dir, "DA2b" + FORMAT_7)
    img_d7 = cv2.imread(img_name)[:, :, 0].copy()
    plt.subplot(4, 3, 7)
    plt.imshow(img_d7)
    plt.axis('off')
    plt.title("Depth Anything v2 Base", fontsize=12)

    # 8 = Depth Anything v2 Large
    img_name = os.path.join(img_dir, "DA2l" + FORMAT_8)
    img_d8 = cv2.imread(img_name)[:, :, 0].copy()
    plt.subplot(4, 3, 8)
    plt.imshow(img_d8)
    plt.axis('off')
    plt.title("Depth Anything v2 Large", fontsize=12)

    # 9 = Depth Anything v2 Small
    img_name = os.path.join(img_dir, "DA2s" + FORMAT_9)
    img_d9 = cv2.imread(img_name)[:, :, 0].copy()
    plt.subplot(4, 3, 9)
    plt.imshow(img_d9)
    plt.axis('off')
    plt.title("Depth Anything v2 Small", fontsize=12)

    # 10 = Zoe Depth
    img_name = os.path.join(img_dir, "zoe" + FORMAT_10)
    img_d10 = cv2.imread(img_name, cv2.IMREAD_UNCHANGED)
    # transformations for zoe depth:
    #   1. invert the depth map
    #   2. normalize the depth map to [0, 1]
    img_d10 = inverse_depth(img_d10)
    img_d10 = cv2.normalize(img_d10, None, 0, 1, cv2.NORM_MINMAX)
    # display
    plt.subplot(4, 3, 10)
    plt.imshow(img_d10)
    plt.axis('off')
    plt.title("Zoe Depth", fontsize=12)

    plt.tight_layout()
    plt.show()

if __name__ == '__main__':

    # parse
    parser = argparse.ArgumentParser()
    parser.add_argument("-i", "--img_dir", type=str,
                        required=True, help="Name of the directory containing the images")
    args, _ = parser.parse_known_args()
    read_images_data(args.img_dir)
        
