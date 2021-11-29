import argparse
import json
import logging
import re
import shutil
import time
from pathlib import Path

import torch
import torch.cuda

from .checkpoint import BEST_MODEL_FILENAME, CHECKPOINT_FILENAME
from .config import get_config
from .logging import init_logging, log_code_version
from .distributed import global_distributed_init

logger = logging.getLogger(__name__)


def get_timestamp(fmt: str = '%Y%m%d_%H%M%S') -> str:
    timestamp = time.strftime(fmt, time.localtime())
    return timestamp


def get_argparser():
    parser = argparse.ArgumentParser()
    parser.add_argument('-c', '--config', required=True,
                        help='Path to config file')
    parser.add_argument('-x', '--ext-config', nargs='*',
                        default=[], help='Extra jsonnet config')
    parser.add_argument('-e', '--experiment-dir', required=True,
                        const=Path('temp') / get_timestamp(),
                        nargs=argparse.OPTIONAL,
                        type=Path,
                        help='Used to keep checkpoint, tensorboard event log, etc.')
    parser.add_argument('--continue', action='store_true', dest='cont')
    parser.add_argument('--load-checkpoint', '--cp', default=None,
                        type=Path,
                        help='Continue from the specified checkpoint')
    parser.add_argument('--debug', action='store_true')
    parser.add_argument('--force', action='store_true')
    parser.add_argument('--run-dir',
                        required=False,
                        type=Path,
                        help='Used to keep log file, code back etc. We will automatically create one if not specified.')
    parser.add_argument('--gpus', nargs='*', default=None)
    parser.add_argument('--cpu', action='store_true')
    parser.add_argument('--test', action='store_true', help='Only run test')
    return parser


RUN_DIR_NAME_REGEX = re.compile('^run_(\d+)_?')


def resolve_runtime_env(args):
    if not args.cpu:
        if not torch.cuda.is_available():
            try:
                torch.cuda.init()
            except Exception as ex:
                raise RuntimeError(
                    'CUDA is not available. To use CPU, pass "--cpu"') from ex
            assert False, "CUDA is not available, but init succeed. shouldn't be possible."

        device_count = torch.cuda.device_count()
        if args.gpus is None:
            args.gpus = list(range(device_count))

        if any(int(gpu) >= device_count for gpu in args.gpus):
            raise ValueError(
                f'GPU {",".join(args.gpus)} requested, but only {device_count} GPUs available.')

    if (not (args.cont or args.test) and
            args.experiment_dir is not None and
            args.experiment_dir.exists() and
            (args.experiment_dir / CHECKPOINT_FILENAME).exists()):
        if not args.force:
            raise RuntimeError(
                f'Experiment directory {args.experiment_dir} exists and contains previous checkpoint. '
                'Pass "--force" to continue'
            )

    if args.cont and not args.load_checkpoint:
        expected_cp = args.experiment_dir / CHECKPOINT_FILENAME
        if not expected_cp.exists():
            raise RuntimeError(f'Attempt to continue but checkpoint {expected_cp} does not exists')
        args.load_checkpoint = expected_cp

    if args.test and not args.load_checkpoint:
        expected_cp = args.experiment_dir / BEST_MODEL_FILENAME
        if not expected_cp.exists():
            raise RuntimeError(f'Attempt to test but checkpoint {expected_cp} does not exists')
        args.load_checkpoint = expected_cp

    if args.experiment_dir is not None and args.run_dir is None:
        run_id = -1
        if args.experiment_dir.exists():
            for previous_runs in args.experiment_dir.iterdir():
                match = RUN_DIR_NAME_REGEX.match(previous_runs.name)
                if match is not None:
                    run_id = max(int(match.group(1)), run_id)
        run_id += 1
        args.run_dir = args.experiment_dir / f'run_{run_id}'

    if args.run_dir.exists():
        if args.force:
            print(f'WARNING: Run directory {args.run_dir} exists.')
        else:
            raise RuntimeError(
                f'Run directory {args.run_dir} exists. '
                'Pass "--force" to continue'
            )

    config = get_config(args)
    args.run_dir.mkdir(parents=True, exist_ok=True)

    init_logging(args)

    if not args.cpu:
        global_distributed_init(args)

    logger.info("Resolved arguments: %s", args)
    logger.info("Resolved config: %s", json.dumps(config, indent=4))
    log_code_version()

    return args, config
