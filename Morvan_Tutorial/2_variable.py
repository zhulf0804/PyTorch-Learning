#coding=utf-8

from __future__ import print_function
from __future__ import division

import torch
import numpy as np

from torch.autograd import Variable

tensor = torch.FloatTensor([[1, 2], [3, 4]])
variable = Variable(tensor, requires_grad=True)

print(tensor)
print(variable)

t_out = torch.mean(tensor*tensor) # x^2
v_out = torch.mean(variable*variable)

print(t_out)
print(v_out)

v_out.backward()
# v_out = 1/4 * sum(var*var)
# d(v_out)/d(var) = 1/4 * 2 * var = 1/2 * var

print(variable.grad)
print(variable.data)
print(variable.data.numpy())
print(variable)









