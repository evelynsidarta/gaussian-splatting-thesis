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

def read_images_data(img_dir):
    
    # 1 = ground truth
    img_name = os.path.join(img_dir, "gt" + FORMAT_1)
    img_d1 = cv2.imread(img_name)[:, :, 0].copy()
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
    img_d10 = cv2.normalize(img_d10, None, 0, 255, cv2.NORM_MINMAX)
    img_d10 = np.uint8(img_d10)
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
        
