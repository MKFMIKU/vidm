DATASETS = 'data/CLEVRER'
EXPERIMENTS = 'experiment'
import os
import numpy as np
from tqdm import tqdm

import torch
import torch.nn.functional as F
import torchvision.transforms.functional as TF
from torchvision.utils import make_grid
from torchvision.transforms.functional import to_pil_image

from guided_diffusion.script_util import create_gaussian_diffusion

from vidm.network import ComplexUModel as MotionModel
from vidm.dataset import ImageFolderDataset

import PIL


resolution = 128

device = "cuda:0"

eval_dataset = ImageFolderDataset(
    path=os.path.join(DATASETS, 'val'),
    nframes=128,
    train=False,
    resolution=resolution,
    use_labels=True,
    xflip=False,
    return_vid=True
)
eval_iterator = iter(torch.utils.data.DataLoader(eval_dataset, num_workers=1, batch_size=1, shuffle=False))

batch = next(iter(eval_iterator))

imgs = np.uint8(255 * (batch[0].transpose(0,1).transpose(1,2).transpose(2,3).cpu().numpy() / 255.))
imgs = [PIL.Image.fromarray(img, 'RGB') for img in imgs]
imgs[0].save('results/clevrer_sample.gif', quality=100, save_all=True, append_images=imgs[1:], duration=100, loop=1)

motion_model = MotionModel(
    image_size=resolution,
    in_channels=3,
    model_channels=128,
    out_channels=6,
    num_res_blocks=2,
    attention_resolutions=[16],
    channel_mult=(1, 1, 2, 2, 4, 4),
    num_heads=4,
    num_head_channels=64,
    resblock_updown=True,
    use_scale_shift_norm=True,
    diffusion_timesteps=1000,
    video_timesteps=128,
).to(device)

# %%
from collections import OrderedDict

# checkpoint = torch.load('/cis/home/kmei1/experiments/dddpm/taichi/checkpoint_accum_taichi.pth.tar', map_location='cpu')
# new_state_dict = OrderedDict()
# for k, v in checkpoint["ema_state_dict"].items():
#     name = k[7:] # remove `module.`
#     new_state_dict[name] = v
# constent_model.load_state_dict(new_state_dict)

checkpoint = torch.load(os.path.join(EXPERIMENTS, 'dddpm', 'checkpoint_accum_cu_frames_clevrer_robust_280000.pth.tar'), map_location='cpu')
new_state_dict = OrderedDict()
for k, v in checkpoint["ema_state_dict"].items():
    name = k[7:] # remove `module.`
    new_state_dict[name] = v
motion_model.load_state_dict(new_state_dict, strict=False)
print("loading motion", checkpoint['iters'])

# %%
gaussian_diffusion = create_gaussian_diffusion(
    steps=1000,
    learn_sigma=True,
    noise_schedule='linear',
    use_kl=False,
    timestep_respacing='100',
    predict_xstart=False,
    rescale_timesteps=False,
    rescale_learned_sigmas=False,
    p2_gamma=1,
    p2_k=1,
)

# %%
def eval_diffusion(model, gaussian_diffusion, resolution):
    img = torch.randn((1, 3, resolution, resolution), device=device)
    indices = list(range(gaussian_diffusion.num_timesteps))[::-1]

    for i in indices:
        t = torch.tensor([i], device=device)
        with torch.no_grad():
            out = gaussian_diffusion.p_mean_variance(model, img, t)
            noise = torch.randn((1, 3, resolution, resolution), device=device)
            nonzero_mask = (
                (t != 0).float().view(-1, *([1] * (len(img.shape) - 1)))
            )  # no noise when t == 0
            img = out["mean"] + nonzero_mask * torch.exp(0.5 * out["log_variance"]) * noise
    return img

def eval_diffusion_frames(model, gaussian_diffusion, resolution, frames, content):
    video = []

    img1_minus_1 = img0 = content
    frames = list(range(frames))
    for f in tqdm(frames):
        vt = torch.tensor([f], device=device)
        img = torch.randn((1, 3, resolution, resolution), device=device)
        indices = list(range(gaussian_diffusion.num_timesteps))[::-1]
        for i in indices:
            dt = torch.tensor([i], device=device)
            with torch.no_grad():
                out = gaussian_diffusion.p_mean_variance(model, img, dt, model_kwargs={'vt':vt, 'x_mins_1': img1_minus_1, 'x_0': img0})
                noise = torch.randn((1, 3, resolution, resolution), device=device)
                nonzero_mask = (
                    (dt != 0).float().view(-1, *([1] * (len(img.shape) - 1)))
                )  # no noise when t == 0
                img = out["mean"] + nonzero_mask * torch.exp(0.5 * out["log_variance"]) * noise
        video.append(img)
        img1_minus_1 = img
    video = torch.cat(video, dim=0)
    return video

# %%
# cotent = eval_diffusion(constent_model, gaussian_diffusion, 128)
cotent = batch[0].transpose(0,1)[0:1] / 255. * 2. - 1.

# %%
video = eval_diffusion_frames(motion_model, gaussian_diffusion, 128, 128, cotent.to(device))

# %%
plt.figure(figsize=(32, 64))
video_grid = make_grid(video, nrow=8, normalize=True, value_range=(-1, 1))
plt.imshow(to_pil_image(video_grid))
plt.savefig('results/clevrer_result.png')

# %%
imgs = np.uint8(255 * (video.transpose(1,2).transpose(2,3).cpu().numpy() / 2. + .5))
imgs = [PIL.Image.fromarray(img, 'RGB') for img in imgs]
imgs[0].save('results/clevrer_result.gif', quality=100, save_all=True, append_images=imgs[1:], duration=100, loop=1)
