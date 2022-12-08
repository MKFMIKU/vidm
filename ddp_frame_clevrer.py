import os
DATASETS = os.getenv('DATASETS')
EXPERIMENTS = os.getenv('EXPERIMENTS')

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
from torchvision.transforms.functional import to_pil_image

from dddpm.unet import ConstantUModel
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
    model = ConstantUModel(
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

    ema_model = copy.deepcopy(model)
    ema_model.eval()

    # optionally resume from a checkpoint
    if args.resume:
        if os.path.isfile(args.resume):
            print("=> loading checkpoint '{}'".format(args.resume))
            checkpoint = torch.load(args.resume, map_location='cpu')
            args.start_iters = checkpoint["iters"]
            model.load_state_dict(checkpoint["state_dict"], strict=False)
            ema_model.load_state_dict(checkpoint["ema_state_dict"], strict=False)
            print(
                "=> loaded checkpoint '{}' (iterations {})".format(
                    args.resume, checkpoint["iters"]
                )
            )
        else:
            print("=> no checkpoint found at '{}'".format(args.resume))
    
    cudnn.benchmark = True

    eval_gaussian_diffusion = create_gaussian_diffusion(
        steps=1000,
        learn_sigma=True,
        noise_schedule='linear',
        use_kl=False,
        timestep_respacing="ddim100",
        predict_xstart=False,
        rescale_timesteps=False,
        rescale_learned_sigmas=False,
        p2_gamma=1,
        p2_k=1,
    )

    results = []
    for iter in tqdm(range(2048 // args.batch_size // torch.cuda.device_count())):
        device = args.gpu
        batch_size = args.batch_size
        resolution=args.resolution

        img = eval_gaussian_diffusion.p_sample_loop(ema_model, (batch_size, 3, resolution, resolution))
        img_list = [img.clone() for _ in range(dist.get_world_size())]
        dist.all_gather(img_list, img)
        img = torch.cat(img_list, dim=0)

        results.append(img / 2 + .5)

    if args.rank % torch.cuda.device_count() == 0:
        results_list = torch.cat(results, dim=0)
        print("gather", results_list.shape)
        results_list = results_list.cpu()
        for idx, result in tqdm(enumerate(results_list)):
            img = to_pil_image(result)
            img.save('frame_results/%05d.png' % idx)
        print("finished")

if __name__ == "__main__":
    main()
