#coding=utf-8

from __future__ import print_function
from __future__ import division

import torch

print(torch.__version__)
print('gpu:', torch.cuda.is_available())
