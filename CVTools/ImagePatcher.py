import os
from os import listdir
from os.path import splitext
from tqdm import tqdm
import cv2
import numpy as np
import sys
import os
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(BASE_DIR)

from GeneralTools.FileOperator import get_files_pth, get_filename_from_pth


def get_files_name(dir_path):
    '''
    返回指定文件夹内的文件名列表
    :param dir_path: 问价夹路径
    :return:指定文件夹内的文件名列表
    '''
    ids = [splitext(file)[0] for file in listdir(dir_path)  # 获取图片名称，ids是一个列表
           if not file.startswith('.')]
    return ids


def presect(dir_path):
    """
    把dir_path下所有的图片进行切割
    :param dir_path:
    :return:
    """
    ids = get_files_name(dir_path)
    length = len(ids)
    for i in tqdm(range(0,length)):
        _overlapping_seg(fr'{dir_path}/{ids[i]}.png',f'{ids[i]}')
    print('done')


def _seg_img(img_path,img_name):
    """
    将任意大小的图片放缩至5376*1408。随后长轴分八份，纵轴分四份进行切割。
    :param img_path: 图片路径
    :param sect_path: 切分后图片存放路径
    """

    img = cv2.imread(img_path)
    img = cv2.resize(img, (5376, 1408))
    w = 672
    h = 352
    for i in range(0, 8):
        for j in range(0, 4):
            roi = img[j * h:(j + 1) * h, i * w:(i + 1) * w]
            retval = cv2.imwrite(f"../data/imgs/{img_name}_{str(i)}_{str(j)}.png", roi)
            assert retval, r"保存失败"

def _smaller_seg_img(img_path,img_name):
    """
    将任意大小的图片放缩至5376*1408。随后长轴分17份，纵轴分7份进行切割。
    :param img_path: 图片路径
    :param sect_path: 切分后图片存放路径
    """

    img = cv2.imread(img_path)
    img = cv2.resize(img, (5440, 1344))
    w = 320
    h = 192
    for i in range(0, 17):
        for j in range(0, 7):
            roi = img[j * h:(j + 1) * h, i * w:(i + 1) * w]
            retval = cv2.imwrite(f"../data/mask_backup/{img_name}_{str(i)}_{str(j)}.png", roi)
            assert retval, r"保存失败"
def _has_no_foreground_information(img:np.ndarray):
    '''
    判断二值图img是否有黑色像素点
    :param img:
    :return:
    '''
    h,w = img.shape
    # print(img.sum()==h*w*255)
    if (h-5)*(w-5)*255<img.sum()<=h*w*255:
        return True
    return False
def delete_NAN_samples(imgs_dir,mask_dir):
    '''
    删除没有前景信息的样本，同时包括图像及其标签
    :return:
    '''
    count = 0
    mask_list = get_files_pth(mask_dir)
    for mask_path in tqdm(mask_list):
        mask = cv2.imread(mask_path,0)
        if _has_no_foreground_information(mask):
            os.remove(os.path.join(imgs_dir,os.path.basename(mask_path)))
            os.remove(os.path.join(mask_dir,os.path.basename(mask_path)))
            count += 1

    print(f'共删除{count}个无效样本')
def _overlapping_seg(img_path,img_name):
    '''
    重叠切片
    :param img_path: 待切图片路径
    :param img_name: 待切图片名称
    :return:
    '''
    img = cv2.imread(img_path)
    h,w,c = img.shape

    patch_h = 256
    patch_w = 256
    stride_h = 128
    stride_w = 128

    n_w = int((w-patch_w)/stride_w)*stride_w+patch_w
    n_h = int((h-patch_h)/stride_h)*stride_h+patch_h

    img = cv2.resize(img, (n_w, n_h))
    n_patch_h = (h-patch_h)//stride_h+1
    n_patch_w = (w-patch_w)//stride_w+1
    n_patches = n_patch_h*n_patch_w

    for i in range(n_patch_w):
        for j in range(n_patch_h):
            y1 = j * stride_h
            y2 = y1 + patch_h
            x1 = i * stride_w
            x2 = x1 + patch_w
            roi = img[y1:y2,x1:x2]
            retval = cv2.imwrite(fr"../data/backup/mask/{img_name}_{str(i)}_{str(j)}.png", roi)
            assert retval, r"保存失败"


# presect(r'../data/source/mask')

# delete_NAN_samples(r'E:\Project\Unet-vanilla\data\dibco',r'E:\Project\Unet-vanilla\data\dibco_thinner_gt')
def recurrennt_seg(img_path,img_name):
    '''
    循环切片
    :param img_path:
    :param img_name:
    :return:
    '''
    patch_size = 256
    img = cv2.imread(img_path,1)
    (h,w,c) = img.shape
    print(h, w, c)
    h_c = (int(h / patch_size) + 1)
    w_c = (int(w / patch_size) + 1)
    print(h_c)
    print(w_c)
    img = cv2.resize(img, (w_c*patch_size, h_c*patch_size))
    for i in range(0, w_c):
        for j in range(0, h_c):
            roi = img[j * patch_size:(j + 1) * patch_size, i * patch_size:(i + 1) * patch_size]
            retval = cv2.imwrite(f"C:/Users/OCEAN\Desktop/103.470/{img_name}_{str(i)}_{str(j)}.png", roi)
            assert retval, r"保存失败"
# recurrennt_seg(r'E:\Project\Unet-vanilla\data\temp-mask\1.png','1')

def overlapping_rec(img):
    '''
    重叠切片
    :param img_path: 待切图片路径
    :param img_name: 待切图片名称
    :return: [子图1,子图2,...,子图N]
    '''
    # print(f'ori input img shape:{img.shape}')
    h,w = img.shape[:2]
    # print(h,w,c)
    patch_h = H
    ratio = patch_h/h
    resized_w = int(w*ratio)
    img = cv2.resize(img, (resized_w, patch_h))
    # print(f'img.shape waiting for overlap resized :{img.shape}')
    h = patch_h
    
    # 不要改动
    patch_w = 512

    stride_w = 256

    # 以长度 patch_h 步长stride_h的方式滑动
    stride_h = H
    # print(img.shape[1],patch_w)

    if patch_w>img.shape[1] and patch_w-img.shape[1] < 30:
        rst = cv2.copyMakeBorder(img,0,0,0,64,cv2.BORDER_CONSTANT,value=(0,0,0))
        rst= cv2.resize(rst,(patch_w,H))
        # print(f'未达到长度-30，直接返回。返回形状:{rst.shape}')
        return [rst]
    if img.shape[1]<patch_w:
        rst = cv2.copyMakeBorder(img,0,0,0,patch_w-img.shape[1],cv2.BORDER_CONSTANT,value=(0,0,0))
        # print(f'未达到长度，直接返回。返回形状:{rst.shape}')
        return [rst]
    # print(ratio)
    # print(img.shape)
    
    # print(f'after copymakeborder img shpae:{img.shape}')
    rescaled_h,rescaled_w = img.shape[:2]
    n_w = int(math.ceil((rescaled_w-patch_w)/stride_w))*stride_w+patch_w
    n_h = H
    
    img = cv2.copyMakeBorder(img,0,0,0,n_w-img.shape[1],cv2.BORDER_CONSTANT,value=(0,0,0))
    # img = cv2.resize(img, (n_w, n_h))
    
    # print(f'长边自适应尺寸:{img.shape}')
    
    rescaled_h,rescaled_w = img.shape[:2]
    n_patch_h = (rescaled_h-patch_h)//stride_h+1
    assert n_patch_h==1,'n_patch_h!=1'
    n_patch_w = (rescaled_w-patch_w)//stride_w+1

    # print(f'n_patch_h：{n_patch_h}，n_patch_w：{n_patch_w}')
    rst = []
    for i in range(n_patch_w):
        x1 = i * stride_w
        x2 = x1 + patch_w
        roi = img[0:H,x1:x2]
        # print(f'roi.shape:{roi.shape}')
        rst.append(roi)
    if len(rst)==0:
        print('overlap len is 0, this means something could be wrong but not that so lethel')
        return [img]
    
    return rst
def merge_str(a:str,b:str,k=2):
    if a != '':
        key = b[1:1+k]
        # print(key)
        index = a.rfind(key) #,len(a)-k-1,len(a)
        # 如果无法合并
        if index == -1:
            # print(f'unable to merge str, return the concat of {a} and {b}')
            rst = a + b #对编辑距离来说 该操作效果更好
        else:
            rst = a[:index]+b[1:]
        return rst
    else:
        return b
def merge_strs(strs:list):
    rst = ''
    for i in strs:
        rst = merge_str(rst,i)
    # if len(strs)>1:
    #     print(strs)
    #     print(f'multiple str merge rst:{rst}')
    return rst