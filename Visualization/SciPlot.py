#科学绘图
import cv2
import numpy as np

from CVTools.CVTools import showim


def mask_diff(pred:np.ndarray,gt:np.ndarray,bb=True):
    '''
    
    :param pred:
    :param gt:
    :param bb:black background，默认是真
    :return:
    '''
    pred = ~pred
    gt = ~gt
    assert pred.size==gt.size, '尺寸不等'
    g = (cv2.bitwise_and(pred,gt))
    r = (pred - gt)
    b = (gt - pred)
    rst = cv2.merge([b,g,r])
    if not bb:
        for i in range(rst.shape[0]):
            for j in range(rst.shape[1]):
                if all(rst[i][j] == np.asarray([0,0,0])):
                    rst[i][j] = [255,255,255]

    return rst
