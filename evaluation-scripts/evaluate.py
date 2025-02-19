import os

def readImages(img_dir, gt_dir, extensionimg = ".png", extensiongt = ".png"):
    images = []
    gts = []

    for idx, fname in enumerate(os.listdir(img_dir)):
        img_name = os.path.join(input, "image" + str(idx) + "0022" + extensionimg)
