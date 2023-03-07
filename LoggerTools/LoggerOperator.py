import csv
import glob
import logging
import os
import argparse
import time
from os.path import splitext
from tqdm import tqdm
import sys
import os
from torch.utils.tensorboard import SummaryWriter
from GeneralTools.Distributed import is_dist_avail_and_initialized
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
# extract_info_from_logs(r'C:\Users\Ocean\Desktop\CAE-logs\0220-COM-NAIVE.txt',r'C:\Users\Ocean\Desktop\CAE-logs\0220-COM-NAIVE.csv','epoch','train_loss','train_loss_main','train_loss_align')

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

import datetime
import os
import time
from collections import defaultdict, deque


import torch
import torch.distributed as dist

class SmoothedValue(object):
    """Track a series of values and provide access to smoothed values over a
    window or the global series average.
    """

    def __init__(self, window_size=20, fmt=None):
        if fmt is None:
            fmt = "{median:.4f} ({global_avg:.4f})"
        self.deque = deque(maxlen=window_size)
        self.total = 0.0
        self.count = 0
        self.fmt = fmt

    def update(self, value, n=1):
        self.deque.append(value)
        self.count += n
        self.total += value * n

    def synchronize_between_processes(self):
        """
        Warning: does not synchronize the deque!
        """
        if not is_dist_avail_and_initialized():
            return
        t = torch.tensor([self.count, self.total], dtype=torch.float64, device='cuda')
        dist.barrier()
        dist.all_reduce(t)
        t = t.tolist()
        self.count = int(t[0])
        self.total = t[1]

    @property
    def median(self):
        d = torch.tensor(list(self.deque))
        return d.median().item()

    @property
    def avg(self):
        d = torch.tensor(list(self.deque), dtype=torch.float32)
        return d.mean().item()

    @property
    def global_avg(self):
        return self.total / self.count

    @property
    def max(self):
        return max(self.deque)

    @property
    def value(self):
        return self.deque[-1]

    def __str__(self):
        return self.fmt.format(
            median=self.median,
            avg=self.avg,
            global_avg=self.global_avg,
            max=self.max,
            value=self.value)

class MetricLogger(object):
    def __init__(self, delimiter="\t"):
        self.meters = defaultdict(SmoothedValue)
        self.delimiter = delimiter

    def update(self, **kwargs):
        for k, v in kwargs.items():
            if v is None:
                continue
            if isinstance(v, torch.Tensor):
                v = v.item()
            assert isinstance(v, (float, int))
            self.meters[k].update(v)

    def __getattr__(self, attr):
        if attr in self.meters:
            return self.meters[attr]
        if attr in self.__dict__:
            return self.__dict__[attr]
        raise AttributeError("'{}' object has no attribute '{}'".format(
            type(self).__name__, attr))

    def __str__(self):
        loss_str = []
        for name, meter in self.meters.items():
            loss_str.append(
                "{}: {}".format(name, str(meter))
            )
        return self.delimiter.join(loss_str)

    def synchronize_between_processes(self):
        for meter in self.meters.values():
            meter.synchronize_between_processes()

    def add_meter(self, name, meter):
        self.meters[name] = meter

    def log_every(self, iterable, print_freq, header=None):
        i = 0
        if not header:
            header = ''
        start_time = time.time()
        end = time.time()
        iter_time = SmoothedValue(fmt='{avg:.4f}')
        data_time = SmoothedValue(fmt='{avg:.4f}')
        space_fmt = ':' + str(len(str(len(iterable)))) + 'd'
        log_msg = [
            header,
            '[{0' + space_fmt + '}/{1}]',
            'eta: {eta}',
            '{meters}',
            'time: {time}',
            'data: {data}'
        ]
        if torch.cuda.is_available():
            log_msg.append('max mem: {memory:.0f}')
        log_msg = self.delimiter.join(log_msg)
        MB = 1024.0 * 1024.0
        for obj in iterable:
            data_time.update(time.time() - end)
            yield obj
            iter_time.update(time.time() - end)
            if i % print_freq == 0 or i == len(iterable) - 1:
                eta_seconds = iter_time.global_avg * (len(iterable) - i)
                eta_string = str(datetime.timedelta(seconds=int(eta_seconds)))
                if torch.cuda.is_available():
                    print(log_msg.format(
                        i, len(iterable), eta=eta_string,
                        meters=str(self),
                        time=str(iter_time), data=str(data_time),
                        memory=torch.cuda.max_memory_allocated() / MB))
                else:
                    print(log_msg.format(
                        i, len(iterable), eta=eta_string,
                        meters=str(self),
                        time=str(iter_time), data=str(data_time)))
            i += 1
            end = time.time()
        total_time = time.time() - start_time
        total_time_str = str(datetime.timedelta(seconds=int(total_time)))
        print('{} Total time: {} ({:.4f} s / it)'.format(
            header, total_time_str, total_time / len(iterable)))
        
        

import json
def parser_persistance(saved_dir,parse_args:argparse.Namespace,):
# 将Namespace对象转换为JSON字符串
    json_string = json.dumps(vars(parse_args))

    # 将JSON字符串写入文件
    with open(os.path.join(saved_dir,'saved_args.txt'), 'w') as file:
        file.write(json_string)
        
def parser_loader(file_pth):
    # 从文件中读取JSON字符串
    with open(file_pth, 'r') as file:
        json_string = file.read()

    # 将JSON字符串转换回Namespace对象
    args_dict = json.loads(json_string)
    args = argparse.Namespace(**args_dict)

    # 输出Namespace对象
    return args