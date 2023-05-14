# Copyright (c) OpenMMLab. All rights reserved.
import argparse
import os.path as osp
import time
from functools import partial
from typing import Optional

import matplotlib.pyplot as plt
from tqdm import tqdm
import numpy as np
import torch
import torch.nn.functional as F

from sklearn.manifold import TSNE
from torch.utils.data import DataLoader

import torch
import torch.nn as nn
from functools import partial
import timm.models.vision_transformer
from sklearn.preprocessing import StandardScaler

from torch.utils.data import DataLoader
import torchvision.datasets as datasets

import os
def auto_make_directory(dir_pth: str):
    '''
    自动检查dir_pth是否存在，若存在，返回真，若不存在创建该路径，并返回假
    :param dir_pth: 路径
    :return: bool
    '''
    if os.path.exists(dir_pth):  ##目录存在，返回为真
        return True
    else:
        os.makedirs(dir_pth)
        return False
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
def parse_args():
    parser = argparse.ArgumentParser(description='t-SNE visualization')
    parser.add_argument('--work-dir', default='',help='the dir to save logs and models')
    parser.add_argument('--exp_name', default='test')
    parser.add_argument('--weights_pth', default=None)
    parser.add_argument('--ckpt_only_weights', action='store_true',)
    parser.add_argument(
        '--max-num-class',
        type=int,
        default=20,
        help='the maximum number of classes to apply t-SNE algorithms, now the'
        'function supports maximum 20 classes')
    parser.add_argument('--seed', type=int, default=0, help='random seed')
    parser.add_argument(
        '--deterministic',
        action='store_true',
        help='whether to set deterministic options for CUDNN backend.')
    parser.add_argument(
        '--device', default='cuda:0', help='Device used for inference')
    # t-SNE settings
    parser.add_argument(
        '--n-components', type=int, default=2, help='the dimension of results')
    parser.add_argument(
        '--perplexity',
        type=float,
        default=30.0,
        help='The perplexity is related to the number of nearest neighbors'
        'that is used in other manifold learning algorithms.')
    parser.add_argument(
        '--early-exaggeration',
        type=float,
        default=12.0,
        help='Controls how tight natural clusters in the original space are in'
        'the embedded space and how much space will be between them.')
    parser.add_argument(
        '--learning-rate',
        type=float,
        default=200.0,
        help='The learning rate for t-SNE is usually in the range'
        '[10.0, 1000.0]. If the learning rate is too high, the data may look'
        'like a ball with any point approximately equidistant from its nearest'
        'neighbours. If the learning rate is too low, most points may look'
        'compressed in a dense cloud with few outliers.')
    parser.add_argument(
        '--n-iter',
        type=int,
        default=1000,
        help='Maximum number of iterations for the optimization. Should be at'
        'least 250.')
    parser.add_argument(
        '--n-iter-without-progress',
        type=int,
        default=300,
        help='Maximum number of iterations without progress before we abort'
        'the optimization.')
    parser.add_argument(
        '--init', type=str, default='random', help='The init method')
    args = parser.parse_args()
    return args

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
            norm_layer = kwargs["norm_layer"]
            embed_dim = kwargs["embed_dim"]
            self.fc_norm = norm_layer(embed_dim)

            del self.norm  # remove the original norm

    def forward_features(self, x):
        B = x.shape[0]
        x = self.patch_embed(x)

        cls_tokens = self.cls_token.expand(
            B, -1, -1
        )  # stole cls_tokens impl from Phil Wang, thanks
        x = torch.cat((cls_tokens, x), dim=1)
        x = x + self.pos_embed
        x = self.pos_drop(x)

        for blk in self.blocks:
            x = blk(x)
        from_backbone = x
        if self.global_pool:
            x = x[:, 1:, :].mean(dim=1)  # global pool without cls token
            outcome = self.fc_norm(x)
        else:
            x = self.norm(x)
            outcome = x[:, 0]

        return outcome,from_backbone

    def forward(self, x):
        x,from_backbone = self.forward_features(x)
        x = self.head(x)
        return x,from_backbone



import torch
def main():
    args = parse_args()
    device = torch.device(args.device if torch.cuda.is_available() else "cpu")
    torch.backends.cudnn.benchmark = True
    # work_dir is determined in this priority: CLI > segment in file > filename

    tsne_work_dir = args.work_dir
    auto_make_directory(osp.abspath(tsne_work_dir))


    # patch_size=16, embed_dim=768, depth=12, num_heads=12, mlp_ratio=4, qkv_bias=True,
    # patch_size=16, embed_dim=384, depth=12, num_heads=6, mlp_ratio=4, qkv_bias=True,
    model = VisionTransformer(
        patch_size=16, embed_dim=768, depth=12, num_heads=12, mlp_ratio=4, qkv_bias=True,
        norm_layer=partial(nn.LayerNorm, eps=1e-6))
    msg = load_model(model, args.weights_pth,args.ckpt_only_weights)# ,True for dino
    print(msg)
    model.eval()
    model.to(device)

    # build the dataset


    import torchvision.transforms as transforms
    import PIL
    from timm.data.constants import IMAGENET_DEFAULT_MEAN, IMAGENET_DEFAULT_STD
    transform = transforms.Compose([
    transforms.Resize((224, 224), interpolation=PIL.Image.BICUBIC),
    transforms.ToTensor(),
    transforms.Normalize(mean=IMAGENET_DEFAULT_MEAN, std=IMAGENET_DEFAULT_STD)
    ])


    # Dont change BATCH SIZE
    # dataset_train = ImageListFolder(os.path.join(args.data_path,'train'), transform=transform, ann_file=os.path.join(args.data_path, 'train.txt'))
    dataset_train = ImageListFolder(os.path.join(r'/opt/data/common/ImageNet/ILSVRC2012','val'), transform=transform, ann_file=os.path.join(r'/opt/data/common/ImageNet/ILSVRC2012', 'val(1).txt'))
    # indices = list(range(32*4))
    # dataset_train = torch.utils.data.Subset(dataset_train, indices)
    dataloader = DataLoader(dataset_train,
                            batch_size=8, # according to your device memory
                            shuffle=False)  # Don't forget to seed your dataloader

    features = []
    labels = []
# tqdm(enumerate(train_loader),total =len(train_loader),
# leave = True)
    for idx,(data,lbl) in tqdm(enumerate(dataloader),total =len(dataloader),leave = True):
        with torch.no_grad():
            # preprocess data

            data = data.to(device)
            # extract backbone features
            _,batch_features = model(data)

            # bs, N, D -> bs, D, N
            batch_features = torch.einsum('ijk->ikj', batch_features)[:,:,1:]
            # print(batch_features.shape)
            num_patch = 224//model.patch_embed.patch_size[0]
            batch_features=batch_features.view(batch_features.shape[0],batch_features.shape[1],num_patch,num_patch)
            # bs, D
            batch_features = F.adaptive_avg_pool2d(batch_features, 1).squeeze()
        features.append(batch_features)
        labels.extend(lbl.cpu().numpy())
    features = torch.cat(features,dim=0).cpu().numpy()
    # save it
    auto_make_directory(f'{tsne_work_dir}/{args.exp_name}')
    output_file = f'{tsne_work_dir}/{args.exp_name}/feature.npy'
    np.save(output_file, features)
    
    scaler = StandardScaler()
    features = scaler.fit_transform(features)


    print(features.shape)
    # build t-SNE model
    tsne_model = TSNE(
        n_components=args.n_components,
        perplexity=args.perplexity,
        early_exaggeration=args.early_exaggeration,
        learning_rate=args.learning_rate,
        n_iter=args.n_iter,
        n_iter_without_progress=args.n_iter_without_progress,
        init=args.init)

    result = tsne_model.fit_transform(features)
    res_min, res_max = result.min(0), result.max(0)
    res_norm = (result - res_min) / (res_max - res_min)
    
    plt.figure(figsize=(10, 10))
    plt.scatter(
        res_norm[:, 0],
        res_norm[:, 1],
        alpha=1.0,
        s=15,
        c=labels,
        cmap='tab20')
    plt.savefig(f'{tsne_work_dir}/{args.exp_name}/t-sne.svg')

if __name__ == '__main__':
    main()