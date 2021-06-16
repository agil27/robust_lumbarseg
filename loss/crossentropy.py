import numpy as np
import jittor as jt
from jittor import init
from jittor import nn
from jittor import Module

class CrossEntropyLoss(Module):
    def __init__(self, ignore_index=None):
        self.ignore_index = ignore_index
        self._backend_loss = nn.CrossEntropyLoss(ignore_index=ignore_index) 

    def __call__(self, input, target):
        return self._backend_loss(input, target)

