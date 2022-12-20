# VIDM: Video Implicit Diffusion Models(AAAI 2023)

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
# install the guided_diffusion dependencies according to https://github.com/openai/guided-diffusion

git clone https://github.com/openai/guided-diffusion.git
cd guided-diffusion
pip install -e .
```

## Training

1. To download [CLEVRER dataset](http://clevrer.csail.mit.edu/) and split videos into frames, run

```bash
chmod a+x scripts/download_clevr_dataset.sh
./scripts/download_clevr_dataset.sh

python scripts/preprocess_clevr_dataset.py
```

2. To train VIDM and motion latent encoder with default settings, run

```bash
CUDA_VISIBLE_DEVICES=0,1,2,3 python diffusion_ddp_accum_constant_cu.py --multiprocessing-distributed --world-size 1 --rank 0 --batch-size 48 --workers 24
```

**Note:** The above training script uses all GPUs by default. You can control the to-be-used GPUs by setting the `CUDA_VISIBLE_DEVICES` variable. The `batch-szie` and `workers` are the total numbers, which will be divied by the number of GPUs inner the training script.
