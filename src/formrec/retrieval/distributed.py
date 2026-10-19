"""DDP 도우미: 프로세스 그룹 시작(GPU nccl · CPU 스모크 gloo), 주 프로세스 로그, 기울기 평균."""
import datetime
import os
import time

import torch
import torch.distributed as dist


def setup(cpu, port):
    """→ (device, rank, world). cpu면 1프로세스 gloo(스모크 · 검증용)."""
    if cpu:
        for k, v in (("MASTER_ADDR", "127.0.0.1"), ("MASTER_PORT", str(port)), ("RANK", "0"), ("WORLD_SIZE", "1")):
            os.environ.setdefault(k, v)
        dist.init_process_group("gloo"); dev = torch.device("cpu")
    else:
        dist.init_process_group("nccl", timeout=datetime.timedelta(hours=2))
        lr_ = int(os.environ.get("LOCAL_RANK", 0)); torch.cuda.set_device(lr_); dev = torch.device("cuda", lr_)
    return dev, dist.get_rank(), dist.get_world_size()


def enable_tf32():
    torch.backends.cuda.matmul.allow_tf32 = True; torch.backends.cudnn.allow_tf32 = True; torch.backends.cudnn.benchmark = True


def is_main():
    return not dist.is_initialized() or dist.get_rank() == 0


def log(*s):
    if is_main():
        print(time.strftime("%H:%M:%S"), *s, flush=True)


def average_gradients(params, world):
    """기울기를 한 벡터로 모아 all_reduce 평균. 기울기가 없는 파라미터는 0 기울기를 받는다(옵티마이저가 그 파라미터도 갱신)."""
    grads = [p.grad if p.grad is not None else torch.zeros_like(p) for p in params]
    flat = torch.cat([g.flatten() for g in grads]); dist.all_reduce(flat); flat /= world
    o = 0
    for p, g in zip(params, grads):
        k = g.numel(); g.copy_(flat[o:o + k].view(g.shape)); p.grad = g; o += k


def finish():
    dist.barrier(); dist.destroy_process_group(); os._exit(0)
