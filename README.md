# VIDM: Video Implicit Diffusion Models

[Kangfu Mei](https://kfmei.page/) and [Vishal M. Patel](https://engineering.jhu.edu/vpatel36/vishal-patel/), Johns Hopkins University, MD, USA

[![paper](https://img.shields.io/badge/arXiv-Paper-<COLOR>.svg)](https://arxiv.org/abs/2111.09881)
[![homepage](https://img.shields.io/badge/Project-Page-red)](https://kfmei.page/vidm/)

#### News
- **Nov 19, 2022:** Paper accepted at AAAI 2023 :tada: 

<hr />

> **Abstract:** *Diffusion models have emerged as a powerful generative method for synthesizing high-quality and diverse set of images. In this paper, we propose a video generation method based on diffusion models, where the effects of motion are modeled in an implicit condition manner, i.e. one can sample plausible video motions according to the latent feature of frames. We improve the quality of the generated videos by proposing multiple strategies such as sampling space truncation, robustness penalty, and positional group normalization. Various experiments are conducted on datasets consisting of videos with different resolutions and different number of frames. Results show that the proposed method outperforms the state-of-the-art generative adversarial networkbased methods byasignificant margin in terms of FVDscores as well as perceptible visual quality.* 
<hr />

## Network Architecture
<img src = "https://i.imgur.com/1mxuYjP.png"> 


## Installation
The model is built in PyTorch 1.8.0 with Python3.8 and CUDA11.6.

For installing, follow these intructions
```bash
# get the development directory
git clone git@github.com:MKFMIKU/VIDM.git
cd VIDM

# install the required packages
conda env create -p envs --file environment.yml
conda activate ./envs/

# instal pytorch according to https://pytorch.org/
pip3 install torch torchvision torchaudio

# install the guided_diffusion in its P2-weighting variant according to https://github.com/jychoi118/P2-weighting
git clone https://github.com/jychoi118/P2-weighting.git
cd P2-weighting
pip install -e .
cd ../

# install the mmmmediting according to https://github.com/open-mmlab/mmediting#installation
pip3 install openmim
mim install mmcv-full
git clone https://github.com/open-mmlab/mmediting.git
cd mmediting
pip3 install -e .
cd ../

# install the development directory
pip3 install -e .
```

## Training

1. To download [CLEVRER dataset](http://clevrer.csail.mit.edu/) and split videos into frames, run

```bash
chmod a+x scripts/download_clevr_dataset.sh
./scripts/download_clevr_dataset.sh
```

2. To train VIDM and motion latent encoder with default settings, run

```bash
CUDA_VISIBLE_DEVICES=0,1,2,3 python diffusion_ddp_accum_constant_cu.py --multiprocessing-distributed --world-size 1 --rank 0 --batch-size 48 --workers 24
```

**Note:** The above training script uses all GPUs by default. You can control the to-be-used GPUs by setting the `CUDA_VISIBLE_DEVICES` variable. The `batch-szie` and `workers` are the total numbers, which will be divied by the number of GPUs inner the training script.


## Evaluation

1. Download the pre-trained model and place it in ./pretrained_models/
```bash
curl https://www.cis.jhu.edu/~kmei1/share/vidm/checkpoints/checkpoint_accum_clevrer_robust_400000.pth.tar -o pretrained_models/checkpoint_accum_clevrer_robust_400000.pth.tar
```

2. Generate video contents or using the extracted first video frames

3. Testing
```bash
CUDA_VISIBLE_DEVICES=0,1,2,3 python diffusion_ddp_accum_constant_cu.py --multiprocessing-distributed --world-size 1 --rank 0 --batch-size 48 --workers 24
```

3. Evaluation with FVD metric. It uses the first GPU card as default, which can changed by setting `CUDA_VISIBLE_DEVICES`.
```bash
python evaluate_FVD.py -dir1 path/to/a/ -dir2 path/to/b/ -b 2 -r 256 -n 128 -ns 2038 -i3d ./i3d_torchscript.pt
```