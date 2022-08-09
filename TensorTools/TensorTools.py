import numpy as np
from torch import Tensor
import sys
import os
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(BASE_DIR)

def tensor_to_ndarray(t:Tensor):
    cpu_tensor = t.cpu()
    res = cpu_tensor.detach().numpy()  # 转回numpy
    # print(res.shape)
    res = np.squeeze(res, 1)
    # res = np.swapaxes(res, 0, 2)
    # res = np.swapaxes(res, 0, 1)
    return res