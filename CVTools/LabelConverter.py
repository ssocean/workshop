import argparse
import base64
import json
import os
import os.path as osp
import numpy as np
import cv2
from tqdm import tqdm
import imgviz
import PIL.Image
from labelme.logger import logger
from labelme import utils
import sys
import os
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(BASE_DIR)
from CVTools.CVTools import otsu_bin
from GeneralTools.FileOperator import get_dirs_pth, get_dirs_name, auto_make_directory

parser = argparse.ArgumentParser()
parser.add_argument("-json_file", default=None)
parser.add_argument("-o", "--out", default=None)
args = parser.parse_args()






# def remove_strip(dir):
#     '''
#     去除指定路径dir下文件名中的空格
#     :param dir:
#     :return:
#     '''
#     files = get_files_pth(dir)
#     for pth in files:
#         os.rename(pth,pth.replace(' ',''))
# remove_strip(r'D:\hongpu\data\Linzechen\train_data\img')


def filter():
    '''
    利用集合操作、删除指定文件夹下不存在json文件的图片
    :return:
    '''
    pass


def main():
    '''
    使用方法,cd进入此文件目录下,输入
    python label_converter.py 存放json文件的路径
    结果会在JSON文件同级目录下的result-img文件夹
    注意 1.此方法会消除文件名中间的空格
    2.JSON文件需要保存图像数据
    :return:
    '''
    logger.warning(
        "This script is aimed to demonstrate how to convert the "
        "JSON file to a single image dataset."
    )
    logger.warning(
        "It won't handle multiple JSON files to generate a "
        "real-use dataset."
    )


    json_file = args.json_file
    if json_file is None:
        json_file = r'C:\Users\Ocean\Downloads\train_data_cpdis(1)\train_data_cpdis\json'
    if args.out is None:
        out_dir = osp.basename(json_file).replace(".", "_")
        out_dir = osp.join(osp.dirname(json_file), out_dir)
    else:
        out_dir = args.out

    if not osp.exists(out_dir):
        os.mkdir(out_dir)
    count = os.listdir(json_file)
    for i in tqdm(range(0, len(count))):
        path = os.path.join(json_file, count[i])
        if os.path.isfile(path):
            fname_list = os.path.split(path)[1].split('.')[:-1]
            fname = '.'.join(fname_list).replace(' ', '')
            try:
                data = json.load(open(path))
            except (ValueError, json.decoder.JSONDecodeError):
                logger.warning(
                    f"{path} 数据损坏,读取失败,即将跳过"
                )
                continue
            else:

                imageData = data.get("imageData")

                if not imageData:
                    imagePath = os.path.join(os.path.dirname(json_file), data["imagePath"])
                    with open(imagePath, "r", encoding='UTF-8') as f:
                        imageData = f.read()
                        imageData = base64.b64encode(imageData).decode("utf-8")
                img = utils.img_b64_to_arr(imageData)

                label_name_to_value = {"_background_": 0}
                for shape in sorted(data["shapes"], key=lambda x: x["label"]):
                    label_name = shape["label"]
                    if label_name in label_name_to_value:
                        label_value = label_name_to_value[label_name]
                    else:
                        label_value = len(label_name_to_value)
                        label_name_to_value[label_name] = label_value
                try:
                    lbl, _ = utils.shapes_to_label(
                        img.shape, data["shapes"], label_name_to_value
                    )
                except AssertionError:
                    print(AssertionError)#该异常由标注时Polygon仅包含两个或一个点产生
                    continue

                label_names = [None] * (max(label_name_to_value.values()) + 1)
                for name, value in label_name_to_value.items():
                    label_names[value] = name

                lbl_viz = imgviz.label2rgb(
                    label=lbl, img=imgviz.asgray(img), label_names=label_names, loc="rb"
                )
                out_pth = os.path.join(out_dir, fname)
                auto_make_directory(out_pth)
                PIL.Image.fromarray(img).save(osp.join(out_pth, "img.png"))
                utils.lblsave(osp.join(out_pth, "label.png"), lbl)
                PIL.Image.fromarray(lbl_viz).save(osp.join(out_pth, "label_viz.png"))

                with open(osp.join(out_pth, "label_names.txt"), "w") as f:
                    for lbl_name in label_names:
                        f.write(lbl_name + "\n")

                logger.info("Saved to: {}".format(out_pth))

    # 输出图像文件至上级目录result-img下
    dir_list = get_dirs_pth(json_file)
    dir_name_list = get_dirs_name(json_file)
    assert len(dir_list) == len(dir_name_list), "长度不一致"
    for i in range(len(dir_list)):
        # print(os.path.join(dir_list[i], 'label.png'))
        temp = os.path.abspath(os.path.join(json_file, ".."))
        final_output = os.path.join(temp, 'result-img')
        auto_make_directory(final_output)
        img = cv2.imread(os.path.join(dir_list[i], 'label.png'), 1)  # 彩色读图片
        img = otsu_bin(img)
        img_name = dir_name_list[i].replace('_json', '') + '.png'
        # print(os.path.join(final_output, img_name))
        ret = cv2.imwrite(os.path.join(final_output, img_name), img)
        assert ret, '图片保存失败'


if __name__ == "__main__":
    main()

