import csv
import glob
import logging
import os
import time
from os.path import splitext
from tqdm import tqdm
import sys
import os
from torch.utils.tensorboard import SummaryWriter
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(BASE_DIR)
from GeneralTools.FileOperator import auto_make_directory, write_csv

def init_logger(out_pth: str = 'logs'):
    '''
    初始化日志类
    :param out_pth: 输出路径，默认为调用文件的同级目录logs
    :return: 日志类实例对象
    '''
    # 日志模块
    logger = logging.getLogger(__name__)
    logger.setLevel(level=logging.INFO)
    auto_make_directory(out_pth)
    handler = logging.FileHandler(fr'{out_pth}/{time.strftime("%Y_%b_%d", time.localtime())}_log.txt')

    handler.setLevel(logging.INFO)
    formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s # %(message)s')
    handler.setFormatter(formatter)
    # 输出到控制台
    console = logging.StreamHandler()
    console.setLevel(logging.INFO)
    # 输出到日志
    logger.addHandler(handler)
    logger.addHandler(console)
    '''
    logger = init_logger(r'r')
    logger.info("Start print log") #一般信息
    logger.debug("Do something") #调试显示
    logger.warning("Something maybe fail.")#警告
    logger.info("'key':'value'")
    '''
    return logger

def extract_info_from_logs(log_pth:str,out_pth:str,*args):
    '''
    
    :return:
    '''
    import json
    rst_lst = []
    with open(log_pth, 'r', encoding='utf-8') as file:
        log_lines = file.readlines()
        append_lst = []
        for log_line in log_lines:
            # print(log_line)
            extracted_dct = json.loads(log_line)
            
            # print(dct)
            for key in args:
                append_lst.append(extracted_dct[key])
                # append_dct.update({key:})
            rst_lst.append(append_lst)
            append_lst = []
            
    write_csv(rst_lst,out_pth)
    print(rst_lst)
    return rst_lst
            

# 'epoch','train_loss','test_acc1')
# 'epoch','train_loss','train_loss_main','train_loss_align')
extract_info_from_logs(r'C:\Users\Ocean\Desktop\CAE-logs\0211-COM-SAE.txt',r'C:\Users\Ocean\Desktop\CAE-logs\0211-COM-SAE.csv','epoch','train_loss','train_loss_main','train_loss_align')

def init_tensorboard(out_dir: str = 'logs'):
    if not os.path.exists(out_dir):  ##目录存在，返回为真
        os.makedirs(out_dir)

    writer = SummaryWriter(log_dir=out_dir)
    '''
    https://pytorch.org/docs/stable/tensorboard.html
    writer.
    add_scalar(tag, scalar_value, global_step=None, walltime=None, new_style=False, double_precision=False)
    add_scalars(main_tag, tag_scalar_dict, global_step=None, walltime=None)
    add_image(tag, img_tensor, global_step=None, walltime=None, dataformats='HWC')
    add_images(tag, img_tensor, global_step=None, walltime=None, dataformats='NCHW')
    '''

    #  writer.close()  需在最后关闭
    return writer
def write_csv(rst: list, file_pth: str, overwrite=False):
    '''
    :param rst:形如[('val1', val2),...,('valn', valn)]的列表
    :param file_pth:输出csv的路径
    :return:
    '''
    mode = 'w+' if overwrite else 'a+'
    file = open(file_pth, mode, encoding='utf-8', newline='')

    csv_writer = csv.writer(file)

    csv_writer.writerows(rst)

    file.close()