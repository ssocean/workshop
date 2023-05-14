from torch_cka import CKA

import torch
import torch
from functools import partial

import torch
import torch.nn as nn

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
        B = x.shape[0]
        x = self.patch_embed(x)

        cls_tokens = self.cls_token.expand(B, -1, -1)  # stole cls_tokens impl from Phil Wang, thanks
        x = torch.cat((cls_tokens, x), dim=1)
        x = x + self.pos_embed
        x = self.pos_drop(x)
        layer_features = []
        for blk in self.blocks:
            x = blk(x)
            layer_features.append(x)


        # if self.global_pool:
        #     x = x[:, 1:, :].mean(dim=1)  # global pool without cls token
        #     outcome = self.fc_norm(x)
        # else:
        #     x = self.norm(x)
        #     outcome = x[:, 0]

        return x,layer_features

    def forward(self, x):
        x,layer_features = self.forward_features(x)
        # x = self.head(x)
        return x#,layer_features

def vit_small_patch16_4heads(**kwargs):
    model = VisionTransformer(
        patch_size=16, embed_dim=384, depth=12, num_heads=6, mlp_ratio=4, qkv_bias=True,
        norm_layer=partial(nn.LayerNorm, eps=1e-6), **kwargs)
    return model
# patch_size=16, embed_dim=768, depth=12, num_heads=12, mlp_ratio=4, qkv_bias=True,
model_1 = VisionTransformer(
    patch_size=16, embed_dim=384, depth=12, num_heads=6, mlp_ratio=4, qkv_bias=True,
    norm_layer=partial(nn.LayerNorm, eps=1e-6))
msg = load_model(model_1, r"/opt/data/private/zph/MAE-main/output/official_released/mocos-300ep.pth")# ,True for dino

model_2 = VisionTransformer(
    patch_size=16, embed_dim=384, depth=12, num_heads=6, mlp_ratio=4, qkv_bias=True,
    norm_layer=partial(nn.LayerNorm, eps=1e-6))
msg = load_model(model_2, r"/opt/data/private/zph/MAE-main/output/finetune/IN1k/230417-mocos-woLD-300ep/checkpoint-99.pth")

#



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
device = torch.device("cuda:5" if torch.cuda.is_available() else "cpu")
# data_dir = r'D:\Data\tiny-imagenet-200\tiny-imagenet-200'
# dataset_train = TinyImageNet(data_dir, train=True,transform=transform)
# dataset_train = MyDataset(r'/opt/data/common/ImageNet/ILSVRC2012/train/n01443537')
# dataset_val = TinyImageNet(data_dir, train=False,transform=transform)

# dataset_train = ImageListFolder(os.path.join(args.data_path,'train'), transform=transform, ann_file=os.path.join(args.data_path, 'train.txt'))
dataset_train = ImageListFolder(os.path.join(r'/opt/data/common/ImageNet/ILSVRC2012','val'), transform=transform, ann_file=os.path.join(r'/opt/data/common/ImageNet/ILSVRC2012', 'val(1).txt'))
indices = list(range(100))
dataset_train = torch.utils.data.Subset(dataset_train, indices)
dataloader = DataLoader(dataset_train,
                        batch_size=100, # according to your device memory
                        shuffle=False)  # Don't forget to seed your dataloader


cka = CKA(model_1,model_2,
    model1_name="ResNet18",model2_name="ResNet18",device='cuda:5')
cka.compare(dataloader) 

cka.plot_results(save_path='mad_mat.pdf')