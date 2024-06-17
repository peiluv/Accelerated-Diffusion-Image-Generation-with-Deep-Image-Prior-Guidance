# ema.py

# Copyright (c) 2025 Chiang Pei-Heng
# This file is part of a project licensed under the MIT License.

import copy
import torch
from config import get_config

args = get_config(from_cli=False)


class EMA:
    def __init__(self, model, decay=0.999):
        self.ema_model = copy.deepcopy(model)
        self.decay = decay
        self.ema_model.eval()  # EMA model do only inference

    @torch.no_grad()
    def update(self, model):
        for ema_param, param in zip(self.ema_model.parameters(), model.parameters()):
            ema_param.data.mul_(self.decay).add_(param.data, alpha=1 - self.decay)
        # 同步 buffer（如 BN 等）
        for ema_buf, buf in zip(self.ema_model.buffers(), model.buffers()):
            ema_buf.copy_(buf)