import torch
import argparse
import os
from zoedepth.models.builder import build_model
from zoedepth.utils.config import get_config
from zoedepth.utils.misc import save_raw_16bit
from zoedepth.utils.misc import colorize
from PIL import Image

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

def pick_model(arg):
    conf = None
    if arg == "zn":
        conf = get_config("zoedepth", "infer")
    elif arg == "zk":
        conf = get_config("zoedepth", "infer", config_version="kitti")
    elif arg == "znk":
        conf = get_config("zoedepth_nk", "infer")
    else:
        print("Model not found! Please use a valid model")
        return None
    return build_model(conf)

def predict(input, output, model, extension = ".png"):
    
    mtd = model.to(DEVICE)
    directory = os.fsencode(input)
    os.makedirs(output, exist_ok = True)

    for idx, files in enumerate(os.listdir(directory)):
        print("\nStarting predictions for input " + input + ":")
        img_name = os.path.join(input, "image" + str(idx) + "0022" + extension)
        img = Image.open(img_name).convert("RGB") # load the image
        depth_numpy = mtd.infer_pil(img)
        fpath = os.path.join(output, "image" + str(idx) + "0022" + extension)
        fpathcol = os.path.join(output, "image" + str(idx) + "0022_colorized" + extension)
        save_raw_16bit(depth_numpy, fpath)
        colored = colorize(depth_numpy)
        Image.fromarray(colored).save(fpathcol) # save colored output

# command line argument parser
if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("-i", "--input", type=str,
                        required=True, help="Name of the folder containing all the images to be processed")
    parser.add_argument("-m", "--model", type=str,
                        required=False, default="zoedepth", help="Pretrained model to use for the prediction. If not set, default resource from model config is used.")
    parser.add_argument("-o", "--output", type=str, 
                        required=True, help="output folder for the resulting images")
    
    args, unknown_args = parser.parse_known_args()
    cur_model = pick_model(args.model)

    if cur_model == None:
        print("\nTerminated.")
    else:
        predict(args.input, args.output, cur_model) # begin depth prediction