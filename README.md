How to Use 3DGS Training Template in Google Colab:
1. upload training_template.ipynb to somewhere in your Google Drive
2. right click and select Open With -> Google Colaboratory (make sure to get Google Colaboratory beforehand)

How to use adjusted codes:
clone the respective original implementation (see below in 'Main Sources' list) from the respective repositories, replace the changed files (from each xxxx-changed folders) from this repository into the cloned repository.
Example: if you wish to use the adjusted files for 2DGS implementation, clone the original 2DGS repository first and then replace the "train.py" and "dataset_readers.py" files in the cloned repository with the code from this repository (in "2DGS-changed" folder).
Warning: do not replace entire folders with the folder found in this repository, simply go inside the folder and change only the python files.

Main Sources:
1. Original Implementation of Gaussian Splatting - https://github.com/graphdeco-inria/gaussian-splatting
2. Colab Template for Gaussian Splatting
      - https://github.com/camenduru/gaussian-splatting-colab
      - https://github.com/benyoon1/gaussian-splat-colab
3. Original Implementation for 2DGS - https://github.com/hbb1/2d-gaussian-splatting
4. Original Implementation for GS2Mesh - https://github.com/yanivw12/gs2mesh/tree/main
5. Original Implementation for RaDe-GS - https://github.com/BaowenZ/RaDe-GS
6. Original Implementation for ZoeDepth - https://github.com/isl-org/ZoeDepth
7. Original Implementation for DepthAnything v1 - https://github.com/LiheYoung/Depth-Anything
8. Original Implementation for DepthAnything v2 - https://github.com/DepthAnything/Depth-Anything-V2

Disclaimer: I only mostly compiled and adapted the code from (2) to run on the Nov 2024 version of Gaussian Splatting from the original repo and streamlined the whole process for better ease of use (for my personal use).
