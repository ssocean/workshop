import torch
import torch.nn as nn
from functools import partial
import timm.models.vision_transformer


from torch.utils.data import DataLoader
import torchvision.datasets as datasets
class ImageListFolder(datasets.ImageFolder):
    from torchvision.datasets.folder import default_loader
    def __init__(self, root, transform=None, target_transform=None,
                 ann_file=None, loader=default_loader):
        self.root = root
        self.transform = transform
        self.loader = loader
        self.target_transform = target_transform
        self.nb_classes = 1000

        assert ann_file is not None
        print('load info from', ann_file)

        self.samples = []
        ann = open(ann_file)
        for elem in ann.readlines():
            cut = elem.split(' ')
            path_current = os.path.join(root, cut[0])
            target_current = int(cut[1])
            self.samples.append((path_current, target_current))
        ann.close()

        print('load finish')
# 定义一个函数来计算两个张量之间的最大范数
def max_norm(tensor1, tensor2):
    return torch.norm(tensor1 - tensor2, p=float('inf')).item()

# 定义一个函数来计算给定层的 Lipschitz 常数
def estimate_lipschitz(layer, num_samples=1000, device='cuda'):
    # 将层移到指定设备
    layer.to(device)

    # 生成随机输入
    input1 = torch.randn(num_samples, *input_shape).to(device)
    input2 = torch.randn(num_samples, *input_shape).to(device)

    # 应用层并计算输出之间的最大范数
    output1 = layer(input1)
    output2 = layer(input2)
    output_norm = max_norm(output1, output2)

    # 计算输入之间的最大范数
    input_norm = max_norm(input1, input2)

    # 计算并返回 Lipschitz 常数的估计值
    return output_norm / input_norm

def load_model(model, weights_pth, only_weights=False):
    checkpoint = torch.load(weights_pth, map_location="cpu")
    # del checkpoint['model']['decoder_pos_embed']
    # del checkpoint['model']['pos_embed']
    if not only_weights:
        msg = model.load_state_dict(checkpoint["model"], strict=False)
    else:
        msg = model.load_state_dict(checkpoint, strict=False)
    print(msg)
    print("Load checkpoint %s" % weights_pth)
    return model

import torch.nn.functional as F
class VisionTransformer(timm.models.vision_transformer.VisionTransformer):
    """ Vision Transformer with support for global average pooling
    img_size=224, patch_size=16, in_chans=3, num_classes=1000, embed_dim=768, depth=12,
                 num_heads=12, mlp_ratio=4., qkv_bias=False, qk_scale=None, drop_rate=0., attn_drop_rate=0.,
                 drop_path_rate=0., hybrid_backbone=None, norm_layer=nn.LayerNorm
    """

    def __init__(self, global_pool=False, **kwargs):
        super(VisionTransformer, self).__init__(**kwargs)

        self.global_pool = global_pool
        if self.global_pool:
            norm_layer = kwargs['norm_layer']
            embed_dim = kwargs['embed_dim']
            self.fc_norm = norm_layer(embed_dim)

            del self.norm  # remove the original norm

    def forward_features(self, x):
        
    

        # disturbed_x = 1e-3 * F.normalize(disturbed_x, dim=1)
        B = x.shape[0]
        x = self.patch_embed(x)

        
        
        cls_tokens = self.cls_token.expand(B, -1, -1)  # stole cls_tokens impl from Phil Wang, thanks
        x = torch.cat((cls_tokens, x), dim=1)
        x = x + self.pos_embed
        x = self.pos_drop(x)
        
        max_liplizs_for_each_layer = []
        delta = 1e-4 * F.normalize(torch.randn(B, x.shape[1],x.shape[2]), dim=(-2,-1)).to(x.device)
        for blk in self.blocks:
            # print(f'input x mean:{torch.mean(x)}')
            
            # print(f'delta mean:{torch.mean(delta)}')
            
            disturbed_x = (x + delta).to(x.device)
            # 计算扰动与未扰动
            x = blk(x)
            # print(f'output x:{torch.mean(x)}')
            # print(f'output x:{torch.norm(x)-torch.norm(disturbed_x),torch.norm(x-disturbed_x)}')
            # print(x.shape)
                    # 添加微小扰动到输入,重新计算输出
            # delta = 1e-3 * F.normalize(torch.randn_like(x), dim=2)
            # delta_norm = torch.norm(delta)
            # normalized_delta = delta / delta_norm
            # delta = 1e-2 * x
            
            # print(f'input disturbed_x mean:{torch.mean(disturbed_x)}')

            disturbed_x = blk(disturbed_x)
            
            # print(f'output disturbed_x mean:{torch.mean(disturbed_x)}')
            
            
            # # 计算输出的范数和扰动后的输出的范数
            # output_norm = x.view(B, -1).norm(dim=1)
            # disturbed_output_norm = disturbed_x.view(B, -1).norm(dim=1)
            
            # 根据公式计算Lipschitz常数
            lipschitz_constant = (disturbed_x - x).view(B, -1).norm(dim=-1) / delta.view(B, -1).norm(dim=-1)
            # print(lipschitz_constant.shape)
            # 取batch size个样本最大值
            max_lipschitz_constant = lipschitz_constant.max().item()
            # print
            max_liplizs_for_each_layer.append(max_lipschitz_constant)


        # if self.global_pool:
        #     x = x[:, 1:, :].mean(dim=1)  # global pool without cls token
        #     outcome = self.fc_norm(x)
        # else:
        #     x = self.norm(x)
        #     outcome = x[:, 0]

        return x,max_liplizs_for_each_layer

    def forward(self, x):
        _,max_liplizs_for_each_layer = self.forward_features(x)
        # x = self.head(x)
        return max_liplizs_for_each_layer

def vit_small_patch16_4heads(**kwargs):
    model = VisionTransformer(
        patch_size=16, embed_dim=384, depth=12, num_heads=6, mlp_ratio=4, qkv_bias=True,
        norm_layer=partial(nn.LayerNorm, eps=1e-6), **kwargs)
    return model

model_1 = VisionTransformer(
    patch_size=16, embed_dim=384, depth=12, num_heads=6, mlp_ratio=4, qkv_bias=True,
    norm_layer=partial(nn.LayerNorm, eps=1e-6))
msg = load_model(model_1, r"/opt/data/private/zph/MAE-main/output/official_released/dino_vitbase16_pretrain.pth",True)# ,True for dino
model_1.eval()




import torchvision.transforms as transforms
import PIL
from timm.data.constants import IMAGENET_DEFAULT_MEAN, IMAGENET_DEFAULT_STD
transform = transforms.Compose([
  transforms.Resize((224, 224), interpolation=PIL.Image.BICUBIC),
  transforms.ToTensor(),
  transforms.Normalize(mean=IMAGENET_DEFAULT_MEAN, std=IMAGENET_DEFAULT_STD)
])

import os
import torch
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# dataset_train = ImageListFolder(os.path.join(args.data_path,'train'), transform=transform, ann_file=os.path.join(args.data_path, 'train.txt'))
dataset_train = ImageListFolder(os.path.join(r'/opt/data/common/ImageNet/ILSVRC2012','val'), transform=transform, ann_file=os.path.join(r'/opt/data/common/ImageNet/ILSVRC2012', 'val(1).txt'))
# indices = list(range(32))
# dataset_train = torch.utils.data.Subset(dataset_train, indices)
dataloader = DataLoader(dataset_train,
                        batch_size=64, # according to your device memory
                        shuffle=False)  # Don't forget to seed your dataloader

from tqdm import tqdm
model_1.to(device)
head_num = 12
layer_num = 12
all_lipliz_for_each_layer = []
with torch.no_grad():
    for idx,(x,_) in tqdm(enumerate(dataloader)):
        print(f'第{idx}批样本：')
        max_liplizs_for_each_layer = model_1(x.to(device))
        all_lipliz_for_each_layer.append(max_liplizs_for_each_layer)
import numpy as np
all_lipliz_for_each_layer = np.array(all_lipliz_for_each_layer,dtype=np.float32)      
lipliz = np.max(all_lipliz_for_each_layer, axis=0)       
print(lipliz)       
        
