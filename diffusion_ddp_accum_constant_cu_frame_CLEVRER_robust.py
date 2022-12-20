DATASETS = 
EXPERIMENTS = os.getenv('EXPERIMENTS')

import os
import builtins
import time
import copy
import random
import warnings
import argparse

import numpy as np
from tqdm import tqdm

import torch
import torch.backends.cudnn as cudnn
import torch.distributed as dist
import torch.multiprocessing as mp
import torch.nn.parallel
import torch.optim as optim
import torch.utils.data
import torch.utils.data.distributed
from torch.utils.tensorboard import SummaryWriter
from torchvision.utils import make_grid

from dddpm.net import ComplexUModel
from dddpm.mydataset import ImageFolderDataset
from guided_diffusion.script_util import create_gaussian_diffusion

parser = argparse.ArgumentParser(description="Major options for PyAnole")
parser.add_argument(
    "--resolution",
    type=int,
    default=128,
    help="resolution of the experiments",
)
parser.add_argument(
    "--seed", type=int, help="seed for initializing training."
)
parser.add_argument("--gpu", default=None, type=int, help="GPU id to use.")
parser.add_argument(
    "--world-size",
    default=-1,
    type=int,
    help="number of nodes for distributed training",
)
parser.add_argument(
    "--rank", default=-1, type=int, help="node rank for distributed training"
)
parser.add_argument(
    "--dist-url",
    default="tcp://localhost:10001",
    type=str,
    help="url used to set up distributed training",
)
parser.add_argument(
    "--dist-backend", default="nccl", type=str, help="distributed backend"
)
parser.add_argument(
    "--multiprocessing-distributed",
    action="store_true",
    help="Use multi-processing distributed training to launch "
    "N processes per node, which has N GPUs. This is the "
    "fastest way to use PyTorch for either single node or "
    "multi node data parallel training",
)
parser.add_argument(
    "--workers",
    default=32,
    type=int,
    metavar="N",
    help="number of data loading workers (default: 32)",
)
parser.add_argument(
    "--batch-size",
    default=256,
    type=int,
    metavar="N",
    help="mini-batch size (default: 256), this is the total "
    "batch size of all GPUs on the current node when "
    "using Data Parallel or Distributed Data Parallel",
)
parser.add_argument(
    "--total-iters", default=800000, type=int, metavar="N", help="number of total iterations to run"
)
parser.add_argument("--start-iters", default=0, type=int, metavar="N",)
parser.add_argument(
    "--print-freq",
    type=int,
    default=100,
    help="frequency of showing training results on console",
)
parser.add_argument(
    "--test-freq",
    type=int,
    default=1000,
    help="frequency of running evaluation",
)
parser.add_argument(
    "--save-freq",
    type=int,
    default=1000,
    help="frequency of running evaluation",
)
parser.add_argument(
    "--resume",
    default="",
    type=str,
    metavar="PATH",
    help="path to latest checkpoint (default: none)",
)
args = parser.parse_args()


def loopy(dl):
    while True:
        for x in iter(dl): yield x


def ema(source, target, decay):
    source_dict = source.state_dict()
    target_dict = target.state_dict()
    for key in target_dict.keys():
        if 'coords' in key:
            continue
        target_dict[key].data.copy_(
            target_dict[key].data * decay +
            source_dict[key].data * (1 - decay))

def main():
    if args.seed is not None:
        random.seed(args.seed)
        np.random.seed(args.seed)
        torch.manual_seed(args.seed)
        cudnn.deterministic = True
        warnings.warn(
            "You have chosen to seed training. "
            "This will turn on the CUDNN deterministic setting, "
            "which can slow down your training considerably! "
            "You may see unexpected behavior when restarting "
            "from checkpoints."
        )

    if args.gpu is not None:
        warnings.warn(
            "You have chosen a specific GPU. This will completely "
            "disable data parallelism."
        )

    if args.dist_url == "env://" and args.world_size == -1:
        args.world_size = int(os.environ["WORLD_SIZE"])

    args.distributed = args.world_size > 1 or args.multiprocessing_distributed

    ngpus_per_node = torch.cuda.device_count()
    if args.multiprocessing_distributed:
        # Since we have ngpus_per_node processes per node, the total world_size
        # needs to be adjusted accordingly
        args.world_size = ngpus_per_node * args.world_size
        # Use torch.multiprocessing.spawn to launch distributed processes: the
        # main_worker process function
        mp.spawn(main_worker, nprocs=ngpus_per_node, args=(ngpus_per_node, args))
    else:
        # Simply call main_worker function
        main_worker(args.gpu, ngpus_per_node, args)


def main_worker(gpu, ngpus_per_node, args):
    args.gpu = gpu

    # suppress printing if not master
    if args.multiprocessing_distributed and args.gpu != 0:

        def print_pass(*args):
            pass

        builtins.print = print_pass

    if args.gpu is not None:
        print("Use GPU: {} for training".format(args.gpu))

    if args.distributed:
        if args.dist_url == "env://" and args.rank == -1:
            args.rank = int(os.environ["RANK"])
        if args.multiprocessing_distributed:
            # For multiprocessing distributed training, rank needs to be the
            # global rank among all the processes
            args.rank = args.rank * ngpus_per_node + gpu
        dist.init_process_group(
            backend=args.dist_backend,
            init_method=args.dist_url,
            world_size=args.world_size,
            rank=args.rank,
        )

    # create model
    print("=> creating model")
    model = ComplexUModel(
        image_size=args.resolution,
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
        spynet_pretrained='spynet_20210409-c6c1bd09.pth'
    )

    if args.distributed:
        # For multiprocessing distributed, DistributedDataParallel constructor
        # should always set the single device scope, otherwise,
        # DistributedDataParallel will use all available devices.
        if args.gpu is not None:
            torch.cuda.set_device(args.gpu)
            model.cuda(args.gpu)
            # When using a single GPU per process and per
            # DistributedDataParallel, we need to divide the batch size
            # ourselves based on the total number of GPUs we have
            args.batch_size = int(args.batch_size / ngpus_per_node)
            args.workers = int((args.workers + ngpus_per_node - 1) / ngpus_per_node)
            model = torch.nn.parallel.DistributedDataParallel(
                model, device_ids=[args.gpu]
            )
        else:
            model.cuda()
            # DistributedDataParallel will divide and allocate batch_size to all
            # available GPUs if device_ids are not set
            model = torch.nn.parallel.DistributedDataParallel(model)
    elif args.gpu is not None:
        torch.cuda.set_device(args.gpu)
        model = model.cuda(args.gpu)
        # comment out the following line for debugging
        raise NotImplementedError("Only DistributedDataParallel is supported.")
    else:
        # AllGather implementation (batch shuffle, queue update, etc.) in
        # this code only supports DistributedDataParallel.
        raise NotImplementedError("Only DistributedDataParallel is supported.")

    optimizer = optim.AdamW(model.parameters(), lr=2e-5)

    ema_model = copy.deepcopy(model)

    # optionally resume from a checkpoint
    if args.resume:
        if os.path.isfile(args.resume):
            print("=> loading checkpoint '{}'".format(args.resume))
            checkpoint = torch.load(args.resume, map_location='cpu')
            args.start_iters = checkpoint["iters"]
            model.load_state_dict(checkpoint["state_dict"], strict=False)
            ema_model.load_state_dict(checkpoint["ema_state_dict"], strict=False)
            optimizer.load_state_dict(checkpoint["optimizer"])
            print(
                "=> loaded checkpoint '{}' (iterations {})".format(
                    args.resume, checkpoint["iters"]
                )
            )
        else:
            print("=> no checkpoint found at '{}'".format(args.resume))
    
    cudnn.benchmark = True

    # Data loading code
    train_dataset = ImageFolderDataset(
        path=os.path.join(DATASETS, 'CLEVRER'),
        nframes=128,
        train=True,
        resolution=args.resolution,
        use_labels=True,
        xflip=True,
    )
    print("Number of datassets", len(train_dataset))

    if args.distributed:
        train_sampler = torch.utils.data.distributed.DistributedSampler(train_dataset)
    else:
        train_sampler = None

    train_loader = loopy(torch.utils.data.DataLoader(
        train_dataset,
        batch_size=args.batch_size,
        shuffle=(train_sampler is None),
        num_workers=args.workers,
        pin_memory=True,
        sampler=train_sampler,
        drop_last=True,
    ))

    if not args.multiprocessing_distributed or (
        args.multiprocessing_distributed and args.rank % ngpus_per_node == 0
    ):
        writer = SummaryWriter(log_dir="runs/accum_p2_weighting_cu_frames_clevrer_robust")
    else:
        writer = None

    # https://github.com/jychoi118/P2-weighting/blob/536a73aacda15a231209f2067238e83f69ac7fcb/guided_diffusion/script_util.py
    gaussian_diffusion = create_gaussian_diffusion(
        steps=1000,
        learn_sigma=True,
        noise_schedule='linear',
        use_kl=False,
        predict_xstart=False,
        rescale_timesteps=False,
        rescale_learned_sigmas=False,
        p2_gamma=1,
        p2_k=1,
    )

    eval_gaussian_diffusion = create_gaussian_diffusion(
        steps=1000,
        learn_sigma=True,
        noise_schedule='linear',
        use_kl=False,
        timestep_respacing="100",
        predict_xstart=False,
        rescale_timesteps=False,
        rescale_learned_sigmas=False,
        p2_gamma=1,
        p2_k=1,
    )

    for iter in range(args.start_iters, args.total_iters):
        if iter % 10000 == 0:
            if args.distributed:
                train_sampler.set_epoch(iter)

        # train for one epoch
        train(train_loader, gaussian_diffusion, optimizer, model, ema_model, iter, writer, args)

        if iter != 0 and iter % args.test_freq == 0 and args.rank % torch.cuda.device_count() == 0:
            eval(train_loader, eval_gaussian_diffusion, ema_model, iter, writer, args)

        if iter != 0 and iter % args.save_freq == 0 and args.rank % torch.cuda.device_count() == 0:
            torch.save({
                "iters": iter,
                "state_dict": model.state_dict(),
                "ema_state_dict": ema_model.state_dict(),
                "optimizer": optimizer.state_dict()
            }, os.path.join(EXPERIMENTS, "dddpm", "checkpoint_accum_cu_frames_clevrer_robust_%06d.pth.tar" % iter))


def train(train_loader, gaussian_diffusion, optimizer, model, ema_model, iter, writer, args):
    batch_time = AverageMeter("Time", ":6.3f")
    data_time = AverageMeter("Data", ":6.3f")
    losses = AverageMeter("Loss", ":.4e")
    progress = ProgressMeter(
        [batch_time, data_time, losses], prefix="Training: "
    )

    device = args.gpu
    
    # measure data loading time
    end = time.time()
    batch = next(train_loader)
    img0, img1, img1_minus_1, target0, target1, frames = batch
    img0 = img0.to(device, non_blocking=True)
    img1 = img1.to(device, non_blocking=True)
    img1_minus_1 = img1_minus_1.to(device, non_blocking=True)
    frame_th = frames.to(device, non_blocking=True).long()
    
    batch = img0.shape[0]
    diffusion_t = np.random.choice(1000, size=(batch,))
    diffusion_t = torch.from_numpy(diffusion_t).to(device).long()
    noise = torch.randn((batch, 3, args.resolution, args.resolution)).to(device)

    data_time.update(time.time() - end)

    optimizer.zero_grad()
    
    loss = gaussian_diffusion.training_losses(model, img1, diffusion_t, model_kwargs={'vt':frame_th, 'x_mins_1': img1_minus_1, 'x_0': img0}, noise=noise)
    loss = torch.mean(loss["loss"])
    loss.backward()
    torch.nn.utils.clip_grad_norm_(model.parameters(), 1.)
    optimizer.step()

    ema(model, ema_model, 0.999)

    losses.update(loss.item(), batch)

    # measure elapsed time
    batch_time.update(time.time() - end)
    end = time.time()

    if iter % args.print_freq == 0 and args.rank % torch.cuda.device_count() == 0:   
        progress.display(iter)
        writer.add_scalar("loss", loss, iter)


def eval(train_loader, gaussian_diffusion, model, iter, writer, args):

    device = args.gpu

    batch = next(train_loader)
    img0, img1, img1_minus_1, target0, target1, frames = batch
    img0 = img0.to(device, non_blocking=True)
    img1 = img1.to(device, non_blocking=True)
    img1_minus_1 = img1_minus_1.to(device, non_blocking=True)
    frame_th = frames.to(device, non_blocking=True).long()
    
    batch = img0.shape[0]
    img = torch.randn((batch, 3, args.resolution, args.resolution), device=device)
    indices = list(range(gaussian_diffusion.num_timesteps))[::-1]

    for i in tqdm(indices):
        t = torch.tensor([i] * batch, device=device)
        with torch.no_grad():
            out = gaussian_diffusion.p_mean_variance(model, img, t, model_kwargs={'vt':frame_th, 'x_mins_1': img1_minus_1, 'x_0': img0})
            noise = torch.randn((batch, 3, args.resolution, args.resolution), device=device)
            nonzero_mask = (
                (t != 0).float().view(-1, *([1] * (len(img.shape) - 1)))
            )  # no noise when t == 0
            img = out["mean"] + nonzero_mask * torch.exp(0.5 * out["log_variance"]) * noise

    video_grid = make_grid(torch.cat([img0, img, img1]), nrow=8, normalize=True, value_range=(-1, 1))

    if args.rank % torch.cuda.device_count() == 0:
        writer.add_image("result", video_grid, iter)


class AverageMeter(object):
    """Computes and stores the average and current value"""

    def __init__(self, name, fmt=":f"):
        self.name = name
        self.fmt = fmt
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count

    def __str__(self):
        fmtstr = "{name} {val" + self.fmt + "} ({avg" + self.fmt + "})"
        return fmtstr.format(**self.__dict__)


class ProgressMeter(object):
    def __init__(self, meters, prefix=""):
        self.meters = meters
        self.prefix = prefix

    def display(self, batch):
        entries = [self.prefix + "[{:07d}]".format(batch)]
        entries += [str(meter) for meter in self.meters]
        print("\t".join(entries))


if __name__ == "__main__":
    main()
