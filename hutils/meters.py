from typing import Optional
import torch
import torch.distributed


class AverageMeter(object):
    """Computes and stores the average and current value"""

    def __init__(self, name, fmt=':f', device=torch.device('cpu')):
        self.name = name
        self.fmt = fmt

        self.val = torch.tensor(0, dtype=torch.float, device=device)
        self.sum = torch.tensor(0, dtype=torch.float, device=device)
        self.count = torch.tensor(0, dtype=torch.int, device=device)

    @torch.no_grad()
    def update(self, val: torch.Tensor, n=1):
        self.val = val
        self.sum += val * n
        self.count += n

    @property
    def avg(self):
        return self.sum / self.count

    def __str__(self):
        fmtstr = '{name} {val' + self.fmt + '} ({avg' + self.fmt + '})'
        return fmtstr.format(
            name=self.name,
            val=self.val.item(),
            avg=self.avg.item(),
        )

    def sync_distributed(self):
        r_count = torch.distributed.all_reduce(self.count, op=torch.distributed.ReduceOp.SUM, async_op=True)
        r_sum = torch.distributed.all_reduce(self.sum, op=torch.distributed.ReduceOp.SUM, async_op=True)
        r_count.wait()
        r_sum.wait()

class GatheringLogitBatch:
    def __init__(self, logits, group):
        self.gathered_logits = [torch.empty_like(logits) for _ in range(torch.distributed.get_world_size(group=group))]
        self.async_op = torch.distributed.all_gather(self.gathered_logits, logits, group=group, async_op=True)
    def wait(self):
        self.async_op.wait()
        gathered_logits = torch.stack(self.gathered_logits, dim=1)
        return gathered_logits.flatten(0,1)

class GatheringLogits:
    '''
    Used together with `torch.utils.data.distributed.DistributedSampler` to gather logits from all ranks
    '''

    def __init__(self, target_device='cpu', group: Optional[torch.distributed.ProcessGroup] = None):
        self.all_logits = []
        self.gathering: Optional[GatheringLogitBatch] = None
        self.target_device = target_device
        self.group = group

    def _commit_gather(self):
        if self.gathering is not None:
            self.all_logits.append(self.gathering.wait().to(device=self.target_device))

    def add(self, logits: torch.Tensor):
        self._commit_gather()
        self.gathering = GatheringLogitBatch(logits, self.group)

    def finish(self, num_samples: int):
        self._commit_gather()
        return torch.cat(self.all_logits)[:num_samples]


