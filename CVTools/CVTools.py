import imp
from operator import le, truediv
import cv2
import numpy as np
import os
from labelme.logger import logger
import os
import sys
import os
from tqdm import tqdm

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(BASE_DIR)
from GeneralTools.FileOperator import auto_make_directory, get_dirs_name, get_dirs_pth
def otsu_bin(img: np.ndarray):
    img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    _, res = cv2.threshold(img, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    return res


def overlapping_seg(img_path: str, rstdir_pth: str, patch_h: int = 256, patch_w: int = 256, stride_h: int = 128,
                    stride_w: int = 128, suffix: str = None):
    '''
    重叠切片
    :param img_path: 待切图片路径
    :param rstdir_pth: 切片存放路径
    :param stride_w: 横方向步长，默认128
    :param stride_h: 纵方向步长，默认128
    :param patch_w: 切片宽，默认256
    :param patch_h: 切片高，默认256
    :param suffix: 切片后缀，默认于原图相等
    :return:切片总数
    '''
    img_info = os.path.split(img_path)[1].split('.')
    img_name = img_info[0]
    img_suffix = img_info[1]
    if suffix:  # 如果给了后缀
        img_suffix = suffix

    img = cv2.imread(img_path)
    h, w, c = img.shape

    n_w = int((w - patch_w) / stride_w) * stride_w + patch_w
    n_h = int((h - patch_h) / stride_h) * stride_h + patch_h

    img = cv2.resize(img, (n_w, n_h))
    n_patch_h = (h - patch_h) // stride_h + 1
    n_patch_w = (w - patch_w) // stride_w + 1
    n_patches = n_patch_h * n_patch_w
    auto_make_directory(rstdir_pth)
    for i in range(n_patch_w):
        for j in range(n_patch_h):
            y1 = j * stride_h
            y2 = y1 + patch_h
            x1 = i * stride_w
            x2 = x1 + patch_w
            roi = img[y1:y2, x1:x2]
            retval = cv2.imwrite(fr"{rstdir_pth}/{img_name}_{str(i)}_{str(j)}.{img_suffix}", roi)
            assert retval, r"image saved failure"
    return n_patches


def showim(img:np.ndarray,img_name:str='image',is_fixed=True):
    '''
    展示图片
    :param img: ndarray格式的图片
    '''
    if is_fixed:
        cv2.namedWindow(img_name, cv2.WINDOW_AUTOSIZE)
    else:
        cv2.namedWindow(img_name, cv2.WINDOW_NORMAL)
    cv2.imshow(img_name, img)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

def get_cnt_corner(cnt):
    leftmost = tuple(cnt[cnt[:,:,0].argmin()][0])
    rightmost = tuple(cnt[cnt[:,:,0].argmax()][0])
    topmost = tuple(cnt[cnt[:,:,1].argmin()][0])
    bottommost = tuple(cnt[cnt[:,:,1].argmax()][0])
    return {'leftmost':leftmost,
            'rightmost':rightmost,
            'topmost':topmost,
            'bottommost':bottommost}
    

def cal_mean_std(images_dir,is_normalized=False):
    """
    给定数据图片根目录,计算图片整体均值与方差
    :param images_dir:
    :return:
    """
    img_filenames = os.listdir(images_dir)
    m_list, s_list = [], []
    for img_filename in tqdm(img_filenames):
        img = cv2.imread(images_dir + '/' + img_filename)
        img = img / 255.0
        m, s = cv2.meanStdDev(img)

        m_list.append(m.reshape((3,)))
        s_list.append(s.reshape((3,)))
        print(m_list)
    m_array = np.array(m_list)
    s_array = np.array(s_list)
    m = m_array.mean(axis=0, keepdims=True)
    s = s_array.mean(axis=0, keepdims=True)

    mean = m[0][::-1] if is_normalized else m[0][::-1]*255
    std = s[0][::-1] if is_normalized else s[0][::-1]*255
    print('mean: ',mean)
    print('std:  ',std)
    return {'mean':mean,'std':std}

def is_color(img):
    '''_summary_
    判断img是否为彩色图像
    Args:
        img (_type_): _description_

    Returns:
        _type_: _description_
    '''
    if isinstance(img, str):
        img = cv2.imread(img)
    elif isinstance(img, np.ndarray):
        pass
    b, g, r = cv2.split(img)
    if np.sum(b) == np.sum(g) == np.sum(r):
        # hist = cv2.calcHist([img],[0],None,[16],[0,256])
        # print(hist)
        return False
    return True
def is_bin_bg_white(img):
    '''_summary_
    判断二值图背景是否为白色
    Args:
        img (_type_): _description_

    Returns:
        _type_: _description_
    '''
    if isinstance(img, str):
        img = cv2.imread(img)
    elif isinstance(img, np.ndarray):
        pass
    h,w,c = img.shape[:2]
    if c!=1:
        img = cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)
    # _, img = cv2.threshold(img, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    max_val = h*w*255
    current_val = np.sum(img)
    ratio = current_val/max_val
    
    if ratio > 0.5:
        return True
    return False
def find_cnt_center(cnt):
    '''_summary_
    计算轮廓cnt的中心坐标
    Args:
        cnt (_type_): _description_

    Returns:
        _type_: _description_
    '''
    M = cv2.moments(cnt) #计算矩特征
    cX = int(M["m10"] / M["m00"])
    cY = int(M["m01"] / M["m00"])
    return (cX,cY)
def extract_roi_by_cnt(img_ori,point):
    img = img_ori.copy()
    poly = np.array(point).astype(np.int32).reshape((-1))
    poly = poly.reshape(-1, 2)
    # 定义四个顶点坐标
    pts = pts.reshape((-1, 1, 2))
    x, y, w, h = cv2.boundingRect(pts)  #轮廓

    # 画多边形 生成mask
    mask = np.zeros(img.shape, np.uint8)
    mask2 = cv2.fillPoly(mask.copy(), [pts],
                            (255, 255, 255))  # 用于求 ROI
    ROI = cv2.bitwise_and(mask2, img)[y:y + h, x:x + w]
    return ROI
def points_to_poly(points):
    poly = np.array(points).astype(np.int32).reshape((-1))
    poly = poly.reshape(-1, 2)
    return [poly.reshape((-1, 1, 2))]
# is_bin_bg_white(r'F:\Data\GJJS-dataset\dataset\train\image-bin\image_21.jpg')
# a = is_color(r'F:\Data\GJJS-dataset\dataset\train\image\image_46.jpg')

# print(a)
# # labelme_to_dataset(r'D:\hongpu\json',r'D:\hongpu\mask')
