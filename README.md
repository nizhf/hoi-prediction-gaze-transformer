# Human-Object Interaction Prediction
## Introduction
The official implementation of our paper "Human-Object Interaction Prediction in Videos through Gaze Following" accepted by CVIU with doi: https://doi.org/10.1016/j.cviu.2023.103741. ArXiv preprint available: https://arxiv.org/abs/2306.03597.
## Our results
On VidHOI (Oracle mode):
|  Future Time  | mAP (QPIC) | Person-wise top-5 Recall | Precision | Accuracy | F1-score|
|:-------------:|:--------------:|:------------------------:|:---------:|:--------:|:-------:|
| 0s (Detection)| 38.61          | 70.91                    | 59.84     | 51.29    | 62.24   |
| 1s            | 37.59          | 72.17                    | 59.98     | 51.65    | 62.78   |
| 3s            | 33.14          | 71.88                    | 60.44     | 52.08    | 62.87   |
| 5s            | 32.75          | 71.25                    | 59.09     | 51.14    | 61.92   |
| 7s            | 31.70          | 70.48                    | 58.80     | 50.56    | 61.36   |

On Action Genome (PredCls mode):
| Rec@10 | Rec@20 | Rec@50 |
|:------:|:------:|:------:|
| 75.4   | 83.7   | 84.3   | 

- [ ] TODO: show qualitative result

## Install
1. Clone the repository recursively:  
`git clone --recurse-submodules https://github.com/nizhf/hoi-prediction-gaze-transformer.git`
2. Create conda environment. We use mamba to accelerate the installation. In addition, as issue [#7](https://github.com/nizhf/hoi-prediction-gaze-transformer/issues/7), opencv in conda-forge seems to be incompatible with torchvision 0.11.0, so we install opencv via pip.  
```
conda install mamba -c conda-forge  # install mamba in base environment
mamba create -n hoi_torch110 python=3.9 -c conda-forge 
conda activate hoi_torch110  
mamba install pytorch==1.10.0 torchvision==0.11.0 torchaudio==0.10.0 torchtext==0.11.0 cudatoolkit=11.3 black Cython easydict gdown imageio ipywidgets matplotlib notebook numpy pandas Pillow PyYAML requests scikit-learn scipy seaborn tqdm tensorboard wandb -c pytorch -c conda-forge
pip install opencv-python
```
3. Our training is using wandb to record training and validation metrics. You may create an account at https://wandb.ai and follow their instruction to login on your PC.   

# Inference on an arbitrary video
1. Some weights can be downloaded automatically. Some weights need to be downloaded manually from [here](https://tumde-my.sharepoint.com/:f:/g/personal/zhifan_ni_tum_de/Ev6sVnE0y2VBnmJ4RD65W7EB7PVDuGQ68Ybkaj31dGCUow?e=t29y5C): all weights in weights/sttrangaze, weights/yolov5/vidor_yolov5l.pt. Put them into weights/... folder in this repo. Also, if automatic download does not work, you can download them from the same link. 
2. Run the run.py script
```
# Detection
python run.py --source path/to/video --out path/to/output_folder --future 0 --hoi-thres 0.3 --print
# For anticipation, set future to 1, 3, 5, or 7
```
This script will create a video with object tracking and gaze following. The HOIs are saved in the output folder and printed in 1 FPS in the console. 

# Train and evaluate the model
Please follow this [instruction](train.md) to prepare the datasets and train our model.

# Citation
If our work is helpful for your research, please consider citing our publication:
```
@article{NI2023103741,
  title = {Human–Object Interaction Prediction in Videos through Gaze Following},
  journal = {Computer Vision and Image Understanding},
  volume = {233},
  pages = {103741},
  year = {2023},
  issn = {1077-3142},
  doi = {https://doi.org/10.1016/j.cviu.2023.103741},
  url = {https://www.sciencedirect.com/science/article/pii/S1077314223001212},
  author = {Zhifan Ni and Esteve {Valls Mascar\'o} and Hyemin Ahn and Dongheui Lee},
}
```
