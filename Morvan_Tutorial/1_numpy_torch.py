#coding=utf-8

from __future__ import print_function
from __future__ import division

import torch
import numpy as np

np_data = np.arange(6).reshape((2, 3))
torch_data = torch.from_numpy(np_data)
tensor2array = torch_data.numpy()

print(np_data)
print(torch_data)
print(tensor2array)

#
data = [-1, -2, 1, 2]
tensor = torch.FloatTensor(data) # 32bit

print(tensor)

tensor = torch.abs(tensor) # 32bit

print(tensor)

tensor = torch.mean(tensor) # 32bit

print(tensor)

data = np.array([[1, 2], [3, 4]])

tensor = torch.FloatTensor(data)

print(np.matmul(data, data))
print(torch.mm(tensor, tensor))

print(data.dot(data))
print(tensor.dot(tensor))


