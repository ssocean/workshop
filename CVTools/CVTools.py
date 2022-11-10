import imp
import math
from operator import le, truediv
import cv2
import numpy as np
import os
from labelme.logger import logger
import os
import sys
import os
from tqdm import tqdm

from TensorTools.TensorTools import ndarray_to_tensor
from GeneralTools.FileOperator import *
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(BASE_DIR)
from GeneralTools.FileOperator import auto_make_directory, get_dirs_name, get_dirs_pth
def otsu_bin(img: np.ndarray):
    if len(img.shape) == 3:
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
    mask2 = cv2.drawContours(mask2,pts,-1,255,thickness=-1)
    ROI = cv2.bitwise_and(mask2, img)[y:y + h, x:x + w]
    return ROI
import imutils
def adapt_rotate(image,angle):
    # image = imutils.resize(image, width=300)
    # 获取图像的维度，并计算中心
    (h, w) = image.shape[:2]
    (cX, cY) = (w // 2, h // 2)
    # 顺时针旋转33度，并保证图像旋转后完整~,确保整个图都在视野范围
    rotated = imutils.rotate_bound(image, angle)
    # showAndWaitKey('rst',rotated)
    return rotated
def get_hor_projection(img_bin):
    img_bin=img_bin
    # showim(img_bin)
    rst = np.sum(img_bin,axis=1)//255
    return rst.tolist()
# is_bin_bg_white(r'F:\Data\GJJS-dataset\dataset\train\image-bin\image_21.jpg')
# a = is_color(r'F:\Data\GJJS-dataset\dataset\train\image\image_46.jpg')
# print(a)
# # labelme_to_dataset(r'D:\hongpu\json',r'D:\hongpu\mask')
def crop_by_hor_projection(hor_projection,threshold):
    '''_summary_
    根据投影信息返回两端第一次非零元素出现位置
    Args:
        hor_projection (_type_): _description_
        threshold (_type_): _description_

    Returns:
        _type_: _description_ top / down
    '''
    l = len(hor_projection)
    top = 0
    down = l
    is_top_clear = False
    is_down_clear = False
    # print(f'threshold is {threshold}')
    # print(hor_projection[-5:])
    #遍历两端
    threshold = 0
    for i in range(l):
        if hor_projection[i]>threshold and not is_top_clear:
            top = i
            is_top_clear = True
        if hor_projection[l-1-i]>threshold and not is_down_clear:
            down = l-1-i
            is_down_clear = True
        if is_top_clear and is_down_clear:
            break
    # print(f'{top,down}/{l}')
    return top,down

def getCorrect1(img):
    '''_summary_
    霍夫变换 要求输入图像为单通道图像
    Args:
        img (_type_): _description_

    Returns:
        _type_: _description_
    '''
    #读取图片，灰度化
    src = img
    # print(src.shape)
    # src = ~src
    # showAndWaitKey("src",src)
    _,bin = cv2.threshold(src, 1, 255, cv2.THRESH_BINARY)


    # showAndWaitKey("gray",gray)
    #腐蚀、膨胀
    # kernel = np.ones((5,5),np.uint8)
    # erode_Img = cv2.erode(bin,kernel)
    # eroDil = cv2.dilate(erode_Img,kernel)
    # showAndWaitKey("eroDil",eroDil)
    #边缘检测
    canny = cv2.Canny(bin,50,150)
    # showAndWaitKey("canny",canny)
    #霍夫变换得到线条
    lines = cv2.HoughLinesP(canny, 1, np.pi / 180, 40,minLineLength=20,maxLineGap=10)
    # drawing = np.zeros(src.shape[:2], dtype=np.uint8)
    #画出线条
    # ks = []
    # thetas = []
    if lines is None:
        return img
    # print(lines.shape)
    # lines.sort(key=dis_btn_points)
    max_dis = 0
    angle = 0
    for line in lines:
        # print(line)
        x1, y1, x2, y2 = line[0]# [[line]]
        r = pow(pow(x2-x1,2)+pow(y2-y1,2),0.5)
        k = float(y1-y2)/(x1-x2)
        theta = np.degrees(math.atan(k))
        # ks.append(k)
        # thetas.append(theta)
        # cv2.line(drawing, (x1, y1), (x2, y2), 255, 1, lineType=cv2.LINE_AA)
        if r>max_dis:
            max_dis = r
            angle = theta
    # showim(drawing)
    # print(thetas)
    theta = -angle

    # print(theta)
    if theta == 0 or abs(theta)>60:
        return img

    rotateImg = adapt_rotate(src,theta)
    # print(rotateImg.shape)

    # showAndWaitKey("rotateImg",rotateImg)
    rotateImg = cv2.cvtColor(rotateImg,cv2.COLOR_BGR2GRAY)
    _,rotateImg_bin = cv2.threshold(rotateImg, 1, 255, cv2.THRESH_BINARY)

    threshold,_ = rotateImg_bin.shape[:2]
    hor_proj = get_hor_projection(rotateImg_bin)
    top,down = crop_by_hor_projection(hor_proj,threshold//20)
    # showAndWaitKey('rst',rotateImg[top:down,:])
    return rotateImg [top:down,:]

def get_first_non_zeros_2D_2(array_2D):
    none_zero_index = (array_2D!=0).argmax(axis=1)
    # first_non_zeros = np.array([array_2D[i,none_zero_index[i]] for i in range(array_2D.shape[0])])
    first_non_zeros = array_2D[range(array_2D.shape[0]),none_zero_index]
    return first_non_zeros

def getCorrect2(img):
    '''_summary_
    基于轮廓的对齐，可用于矫正任意弯曲的图像
    Args:
        img (_type_): _description_

    Returns:
        _type_: _description_
    '''
    h,w = img.shape[:2]
    if len(img.shape)==3:
        rst = np.zeros([h,w,3],dtype=np.uint8)
        img_gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

        _,img_bin = cv2.threshold(img_gray, 1, 255, cv2.THRESH_BINARY)
    else:
        rst = np.zeros([h,w],dtype=np.uint8)
        img_gray = img
        _,img_bin = cv2.threshold(img_gray, 1, 255, cv2.THRESH_BINARY)

    none_zero_index = (img_bin!=0).argmax(axis=0)
    for i,indent in enumerate(none_zero_index):
        if len(img.shape)==3:
            rst[:,i,:] = np.roll(img[:,i,:], -indent,axis=0)
        else:
            rst[:,i] = np.roll(img[:,i], -indent)
    if len(img.shape)==3:
        deskew_gray_rst = cv2.cvtColor(rst, cv2.COLOR_BGR2GRAY)
    else:
        deskew_gray_rst = rst
    _, deskew_bin_rst = cv2.threshold(deskew_gray_rst, 1, 255, cv2.THRESH_BINARY)
    # showim(deskew_rst)
    hor_proj = get_hor_projection(deskew_bin_rst)
    threshold,_ = deskew_bin_rst.shape[:2]
    top,down = crop_by_hor_projection(hor_proj,threshold//20)
    rst = rst[top:down,:]
    return rst
from skimage import util
from PIL import Image
def dir_to_1bit_bin(dir,output_dir):
    fpths = get_files_pth(dir)
    for pth in tqdm(fpths):
        bin = cv2.imread(pth, cv2.IMREAD_GRAYSCALE)
        _, bin = cv2.threshold(bin, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
        # bin = otsu_bin(bin)/255
        bin = util.img_as_bool(bin)
        img = Image.fromarray(bin)
        new_name = get_filename_from_pth(pth, False)
        img.save(os.path.join(output_dir, new_name+'.png'), bits=1, optimize=True)
        
def get_white_ratio(bbox:np.ndarray):
    '''
    针对黑底白字
    '''
    if len(bbox.shape)>2:
        #三通道 转灰度图
        bbox_gray = cv2.cvtColor(bbox,cv2.COLOR_BGR2GRAY)
    else:
        bbox_gray = bbox
    
    _,bbox_bin = cv2.threshold(bbox_gray,0,255,cv2.THRESH_BINARY+cv2.THRESH_OTSU)
    bbox_bin.astype(np.uint16)
    h,w = bbox_bin.shape[:2]

    bbox_bin = bbox_bin/255
    current_val = np.sum(bbox_bin)
    ratio = current_val/(h*w) #
    return ratio
def cv2_chinese_text(img, text, position, textColor=(0, 0, 255), textSize=30):
    if text is None:
        return img
    if (isinstance(img, np.ndarray)):  # 判断是否OpenCV图片类型
        img = Image.fromarray(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
    # 创建一个可以在给定图像上绘图的对象
    draw = ImageDraw.Draw(img)
    # 字体的格式
    fontStyle = ImageFont.truetype(".ttc", textSize, encoding="utf-8")
    # 绘制文本
    draw.text(position, text, textColor, font=fontStyle,direction='ttb')
    # 转换回OpenCV格式
    return cv2.cvtColor(np.asarray(img), cv2.COLOR_RGB2BGR)
def points_to_poly(points):
    poly = np.array(points).astype(np.int32).reshape((-1))
    poly = poly.reshape(-1, 2)
    return [poly.reshape((-1, 1, 2))]
def resize_contour(cnts,ori_size,rst_shape):
        '''
        原地操作函数，由于原图尺寸的变换将会导致标注信息的变换，该方法完成在图片尺寸变换时标注信息的同步转换。
        最好由低分辨率放大至高分辨率
        :return:
        '''
        o_h, o_w = ori_size
        r_h, r_w= rst_shape
        height_ratio = r_h / o_h
        width_ratio = r_w / o_w  # 计算出高度、宽度的放缩比例
        ratio_mat = [[width_ratio,0],[0,height_ratio]]
        # print(points_to_poly(cnts).shape)
        return (np.array(cnts).astype(np.int32).reshape((-1)).reshape((-1,  2))@ratio_mat).astype(np.int32) # n×2 矩阵乘 2×2