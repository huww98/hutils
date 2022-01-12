import logging
import sys
import os
from tqdm import tqdm as _tqdm

import torch.distributed

logger = logging.getLogger(__name__)

TQDM_DEFAULTS = {
    'dynamic_ncols': True
}

def tqdm(*args, **kwargs):
    new_kwargs = {
        'disable': None if not torch.distributed.is_initialized() or torch.distributed.get_rank() == 0 else True,
    }
    new_kwargs.update(TQDM_DEFAULTS)
    new_kwargs.update(kwargs)
    return _tqdm(*args, **new_kwargs)

class TqdmHandler(logging.Handler):
    def emit(self, record):
        try:
            msg = self.format(record)
            _tqdm.write(msg, file=sys.stderr)
            self.flush()
        except (KeyboardInterrupt, SystemExit):
            raise
        except:
            self.handleError(record)

def init_logging(args, local_primary=True, tqdm=False):
    format = '{asctime:s}|{levelname:8s}|{message:s}'
    formatter = logging.Formatter(fmt=format, style='{')

    handlers = []
    if local_primary:
        if tqdm:
            console_handler = TqdmHandler()
        else:
            console_handler = logging.StreamHandler(stream=sys.stderr)
        console_handler.setFormatter(formatter)
        handlers.append(console_handler)

    if args.run_dir is not None:
        filename = 'experiment.log'
        if torch.distributed.is_initialized():
            filename = f'experiment_rank{torch.distributed.get_rank()}.log'
        file_handler = logging.FileHandler(args.run_dir / filename)
        file_handler.setFormatter(formatter)
        handlers.append(file_handler)

    level = logging.DEBUG if args.debug else logging.INFO

    logging.basicConfig(level=level, handlers=handlers, force=True)

def log_runtime_env():
    import socket
    logger.info('Running on host %s', socket.getfqdn())

def log_code_version():
    import subprocess
    try:
        git_desc = subprocess.run(['git', 'describe', '--always', '--dirty', '--long'], capture_output=True, text=True)
    except FileNotFoundError:
        logger.warn('git not installed, will not record code version.')

    if git_desc.returncode != 0:
        logger.warn('git describe failed:\n%s', git_desc.stderr.strip())
    else:
        logger.info('Current code version: %s', git_desc.stdout.strip())
