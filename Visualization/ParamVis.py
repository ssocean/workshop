import torch


def get_param_value_dict(model):
    """
    Parameter groups for layer-wise lr decay
    Following BEiT: https://github.com/microsoft/unilm/blob/master/beit/optim_factory.py#L58
    """
    param_group_names = {}
    param_groups = {}
    values = []
    for name, param in model.named_parameters():
        # param_groups.update({name:[torch.norm(param,p=1).item(),torch.norm(param,p=2).item(),torch.norm(param,p=float('inf')).item()]})
        # print(name)
        if name.startswith("block") and name.endswith('.attn.qkv.weight'):
            param_groups.update({name: torch.norm(param, p=2).item()})
            values.append(torch.norm(param, p=2).item())
        # print(torch.norm(param,p=1).item())
        # print(torch.norm(param,p=2).item())
        # print(torch.norm(param,p=float('inf')).item())

        # param_group_names[group_name]["params"].append(n)
        # param_groups[group_name]["params"].append(p)

    # print("parameter groups: \n%s" % json.dumps(param_group_names, indent=2))

    return param_groups,values


from functools import partial

import torch
import torch.nn as nn

import timm.models.vision_transformer


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

        for blk in self.blocks:
            x = blk(x)

        if self.global_pool:
            x = x[:, 1:, :].mean(dim=1)  # global pool without cls token
            outcome = self.fc_norm(x)
        else:
            x = self.norm(x)
            outcome = x[:, 0]

        return outcome

    def forward(self, x):
        x = self.forward_features(x)
        x = self.head(x)
        return x


def load_model(model, weights_pth):
    checkpoint = torch.load(weights_pth, map_location='cpu')
    # del checkpoint['model']['decoder_pos_embed']
    # del checkpoint['model']['pos_embed']
    msg = model.load_state_dict(checkpoint['model'], strict=False)
    print(msg)
    print("Load checkpoint %s" % weights_pth)
    return model


# weights1_pth = r"C:\Users\Ocean\Downloads\mae_finetuned_vit_base.pth"
# weights2_pth = r"C:\Users\Ocean\Downloads\mae_pretrain_vit_base_full-wdd80.pth"
# weights3_pth = r"C:\Users\Ocean\Downloads\mae_pretrain_vit_base.pth"
pths = [r"C:\Users\Ocean\Downloads\mae_finetuned_vit_base.pth",r"C:\Users\Ocean\Downloads\mae_pretrain_vit_base_full-wdd80.pth",r"C:\Users\Ocean\Downloads\mae_pretrain_vit_base.pth"]
for pth in pths:
    model = VisionTransformer(
        patch_size=16, embed_dim=768, depth=12, num_heads=12, mlp_ratio=4, qkv_bias=True,
        norm_layer=partial(nn.LayerNorm, eps=1e-6))
    msg = load_model(model, pth)
    p, v = get_param_value_dict(model)
    print(p)
    print(v)
# model = VisionTransformer(
#     patch_size=16, embed_dim=768, depth=12, num_heads=12, mlp_ratio=4, qkv_bias=True,
#     norm_layer=partial(nn.LayerNorm, eps=1e-6))
# msg = load_model(model,weights_pth)
# model_2 = VisionTransformer(
#     patch_size=16, embed_dim=768, depth=12, num_heads=12, mlp_ratio=4, qkv_bias=True,
#     norm_layer=partial(nn.LayerNorm, eps=1e-6))
# msg = load_model(model_2,weights2_pth)
# model_3 = VisionTransformer(
#     patch_size=16, embed_dim=768, depth=12, num_heads=12, mlp_ratio=4, qkv_bias=True,
#     norm_layer=partial(nn.LayerNorm, eps=1e-6))
# msg = load_model(model_3,weights3_pth)
# print(msg)
# p,v = get_param_value_dict(model)
# print(p)
# print(v)
# p,v = get_param_value_dict(model_2)
# print(p)
# print(v)
# p,v = get_param_value_dict(model_3)
# print(p)
# print(v)
