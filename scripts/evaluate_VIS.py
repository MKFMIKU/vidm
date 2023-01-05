import argparse
import os
import numpy as np
import scipy.linalg
import torch
from tqdm import tqdm
from vidm.dataset import VideoFolder


def compute_fvd(feats_fake: np.ndarray, feats_real: np.ndarray):
    mu_gen, sigma_gen = compute_stats(feats_fake)
    mu_real, sigma_real = compute_stats(feats_real)

    m = np.square(mu_gen - mu_real).sum()
    s, _ = scipy.linalg.sqrtm(
        np.dot(sigma_gen, sigma_real), disp=False
    )  # pylint: disable=no-member
    fid = np.real(m + np.trace(sigma_gen + sigma_real - s * 2))

    return float(fid)


def compute_stats(feats: np.ndarray):
    mu = feats.mean(axis=0)  # [d]
    sigma = np.cov(feats, rowvar=False)  # [d, d]

    return mu, sigma


if __name__ == "__main__":

    parser = argparse.ArgumentParser()
    parser.add_argument("-dir2", "--dir2", type=str, default="./preds/")
    parser.add_argument("-b", "--batch", type=int, default=32)
    parser.add_argument("-r", "--resolution", type=int, default=256)
    parser.add_argument("-n", "--nframes", type=int, default=128)
    parser.add_argument("-ns", "--nsamples", type=int, default=2048)

    parser.add_argument("-c3d", "--c3d", type=str)
    opt = parser.parse_args()

    device = "cuda:0"
    batch_size = opt.batch
    resolution = opt.resolution
    nframes = opt.nframes
    nsamples = opt.nsamples

    detector = torch.jit.load(opt.c3d).eval().to(device)

    pred_dataset = VideoFolder(path=opt.dir2, nframes=nframes, size=resolution)
    pred_dataset_iter = iter(
        torch.utils.data.DataLoader(
            pred_dataset, num_workers=8, batch_size=batch_size, shuffle=False
        )
    )

    print(f"loading videos with number of {len(pred_dataset)}")

    feats_fake = []
    for i in tqdm(range(nsamples // batch_size)):
        video = next(pred_dataset_iter).to(device)  # b,n,h,w,c => b,c,n,h,w
        video = video.permute(0, 4, 1, 2, 3).contiguous()
        with torch.no_grad():
            micro_feats_real = detector(video).cpu().numpy()
        feats_fake.append(micro_feats_real)
    feats_fake = np.concatenate(feats_fake, axis=0)

    scores = []
    num_gen = nsamples
    num_splits = 10
    for i in range(num_splits):
        part = feats_fake[i * num_gen // num_splits : (i + 1) * num_gen // num_splits]
        kl = part * (np.log(part) - np.log(np.mean(part, axis=0, keepdims=True)))
        kl = np.mean(np.sum(kl, axis=1))
        scores.append(np.exp(kl))

    print(float(np.mean(scores)), float(np.std(scores)))
