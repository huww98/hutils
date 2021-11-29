from pathlib import Path

import torch
import torch.distributed

from .logging import init_logging, log_runtime_env

def init_file_path(args) -> Path:
    return args.run_dir / 'distributed_init'

def global_distributed_init(args):
    init_file_path(args).unlink(missing_ok=True)

def worker_init(args, local_rank: int, tqdm=True):
    torch.distributed.init_process_group(
        backend='nccl',
        init_method=init_file_path(args).resolve().as_uri(),
        world_size=len(args.gpus),
        rank=local_rank,
    )
    device = torch.device(f'cuda:{args.gpus[local_rank]}')
    torch.cuda.set_device(device)

    init_logging(args, tqdm=tqdm, local_primary=(local_rank == 0))
    log_runtime_env()

    return device
