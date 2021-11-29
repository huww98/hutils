import os
import shutil
from pathlib import Path
import logging
from torch import Tensor
from collections import OrderedDict

import torch

logger = logging.getLogger(__name__)

CHECKPOINT_FILENAME = 'checkpoint.pth.tar'
BEST_MODEL_FILENAME = 'model_best.pth.tar'

class CheckpointManager:
    ''' Manage checkpoints

    * keep_interval (int): Keep a checkpoint every `keep_interval` epochs. None means only keep the latest one.
    * save_interval (int): Maximum checkpoint save interval. Set this to larger than 1 to avoid save checkpoint
                           too often in case one epoch is too small.
    '''

    def __init__(self, experiment_dir: Path, keep_interval=None, save_interval=1, milestone=0):
        """

        :param experiment_dir:
        :param keep_interval:
        :param filename:
        :param milestone: 用来控制从哪个epoch开始，进行interval式保存
        """
        self.experiment_dir = experiment_dir
        self.keep_interval = keep_interval
        self.save_interval = save_interval
        self._last_checkpoint_epoch = 0
        self.milestone = milestone

    def save(self, state: dict, is_best: bool, epoch: int):
        need_keep = self.keep_interval is not None and epoch % self.keep_interval == 0 and epoch > self.milestone
        need_save = is_best or need_keep or self._last_checkpoint_epoch + self.save_interval <= epoch

        if not need_save:
            return

        checkpoint_path = self.experiment_dir / CHECKPOINT_FILENAME
        temp_checkpoint_path = self.experiment_dir / f'.next.{CHECKPOINT_FILENAME}'

        logger.info('Saving checkpoint to "%s"', checkpoint_path)
        try:
            # Save to temp file first.
            # In case of error, previous checkpoint should be kept intact.
            torch.save(state, temp_checkpoint_path)
        except:
            if temp_checkpoint_path.exists():
                temp_checkpoint_path.unlink()
            raise
        temp_checkpoint_path.rename(checkpoint_path)
        logger.info('Checkpoint saved')
        self._last_checkpoint_epoch = epoch

        if is_best:
            model_best_path = self.experiment_dir / BEST_MODEL_FILENAME
            logger.info('Saving best model to "%s"', model_best_path)
            if model_best_path.exists():
                model_best_path.unlink()
            os.link(checkpoint_path, model_best_path)

        if need_keep:
            keep_path = self.experiment_dir / f'checkpoint_epoch_{epoch}.pth.tar'
            logger.info('Keep checkpoint "%s"', keep_path)
            os.link(checkpoint_path, keep_path)
