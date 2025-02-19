import torch
import argparse
import os
import OpenEXR
import Imath
import numpy as np
from zoedepth.models.builder import build_model
from zoedepth.utils.config import get_config
from zoedepth.utils.misc import save_raw_16bit
from zoedepth.utils.misc import colorize
from PIL import Image

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

# credits: https://gist.github.com/jadarve/de3815874d062f72eaf230a7df41771b

# retrieve depth and rgb values from an exr file
def readEXR(filename):
    
    exrf = OpenEXR.InputFile(filename)
    header = exrf.header()
    dw = header['dataWindow']
    in_size = (dw.max.y - dw.min.y + 1, dw.max.x - dw.min.x + 1)
    channel_data = dict()

    # convert all channels to numpy arrays
    for c in header['channels']:
        C = exrf.channel(c, Imath.PixelType(Imath.PixelType.FLOAT))
        C = np.fromstring(C, dtype=np.float32)
        C = np.reshape(C, in_size)

        channel_data[c] = C

    color_channels = ['R', 'G', 'B', 'A'] if 'A' in header['channels'] else ['R', 'G', 'B']
    img = np.concatenate([channel_data[c][...,np.newaxis] for c in color_channels], axis=2)

    # linear to standard RGB
    img[..., :3] = np.where(img[..., :3] <= 0.0031308,
                            12.92 * img[..., :3],
                            1.055 * np.power(img[..., :3], 1 / 2.4) - 0.055)
    
    # sanitize image to be in range [0, 1]
    img = np.where(img < 0.0, 0.0, np.where(img > 1.0, 1, img))
    
    Z = None if 'Z' not in header['channels'] else channel_data['Z']
    return img, Z

def colorize_all(input, output, extension_input = ".exr", extension_output = ".png"):

    directory = os.fsencode(input)
    os.makedirs(output, exist_ok = True)

    for idx, _ in enumerate(os.listdir(directory)):
        img_name = os.path.join(input, "depth" + str(idx) + "0022" + extension_input)
        _, depth_numpy = readEXR(img_name)
        fpathcol = os.path.join(output, "image" + str(idx) + "0022_colorized" + extension_output)
        colored = colorize(depth_numpy)
        Image.fromarray(colored).save(fpathcol) # save colored output

# command line argument parser
if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("-i", "--input", type=str,
                        required=True, help="Name of the folder containing all the images to be processed")
    parser.add_argument("-o", "--output", type=str, 
                        required=True, help="output folder for the resulting images")
    
    args, unknown_args = parser.parse_known_args()

    colorize_all(args.input, args.output) # begin depth prediction
    print("All depth maps successfully colorized.")