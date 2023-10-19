from typing import Type
import torch
import torch.nn as nn
from src.resnet import ResNet, BasicBlock
from einops import reduce
from habitat import logger


class FiLM(nn.Module):
    """
    A Feature-wise Linear Modulation Layer from
    'FiLM: Visual Reasoning with a General Conditioning Layer'
    """
    def __init__(self, channels=[32,64,128,256], res=[32,16,8,4], reduction="none"):
        super().__init__()
        self.reduction = reduction
        if self.reduction == "none":
            self.f = nn.ModuleList([
                nn.Conv2d(channels[i], channels[i], kernel_size=1, stride=1)
                for i in range(len(channels))
            ])
            self.h = nn.ModuleList([
                nn.Conv2d(channels[i], channels[i], kernel_size=1, stride=1)
                for i in range(len(channels))
            ])
        else:
            self.f = nn.ModuleList([
                nn.Linear(channels[i], channels[i])
                for i in range(len(channels))
            ])
            self.h = nn.ModuleList([
                nn.Linear(channels[i], channels[i])
                for i in range(len(channels))
            ])
        
    def forward(self, x, conds, i):
        if self.reduction == "none":
            gammas = self.f[i](conds).view_as(x)
            betas = self.h[i](conds).view_as(x)
        elif self.reduction == "global":
            gammas = self.f[i](reduce(conds, "b c h w -> b c", reduction="mean"))[:,:,None,None].expand_as(x)
            betas = self.h[i](reduce(conds, "b c h w -> b c", reduction="mean"))[:,:,None,None].expand_as(x)
        else:
            raise TypeError("not implemented")
        return (gammas * x) + betas


class ConditionResNet(ResNet):
    def forward(self, x):
        x = self.stem(x)

        interm_o = []
        for l in self.layers:
            x = l(x)
            interm_o.append(x)

        return x, interm_o
    

class FiLMedResNet(ResNet):
    def __init__(self, reduction, film_layers, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.film = FiLM(reduction=reduction)
        self.film_layers = film_layers
        logger.info(f"filmed resnet encoder with layers: {film_layers}, reduction: {reduction}")
    
    def forward(self, x, x_cond):
        x = self.stem(x)

        for i,l in enumerate(self.layers):
            x = l(x)
            if i in self.film_layers:
                x = self.film(x, x_cond[i], i)

        return x


class MidFusionResNet(nn.Module):
    def __init__(
        self,
        reduction,
        film_layers,
        *args, **kwargs
    ):
        super().__init__()
        self.stem_o = FiLMedResNet(reduction, film_layers, *args, **kwargs)
        self.stem_g = ConditionResNet(*args, **kwargs)
        self.film_layers = film_layers
        self.final_spatial_compress = self.stem_o.final_spatial_compress
        # self.final_channels = self.stem_o.final_channels * 2
        self.final_channels = self.stem_o.final_channels
        
    def forward(self, x):
        b,c,h,w = x.shape
        x_o = x[:,:3,...]
        x_g = x[:,3:,...]
        x_g, x_cond = self.stem_g(x_g)
        x_o = self.stem_o(x_o, x_cond)

        # return torch.cat((x_o, x_g), dim=1)
        return x_o


def resnet9(in_channels, base_planes, ngroups, film_reduction, film_layers):
    return MidFusionResNet(film_reduction, film_layers, 3, base_planes, ngroups, BasicBlock, [1, 1, 1, 1])


if __name__ == "__main__":
    mid_fusion_resnet = MidFusionResNet(
        3, 32, 16, BasicBlock, [1, 1, 1, 1]
    )
    dummy = torch.rand([2,6,128,128])
    out = mid_fusion_resnet(dummy)
    print(out.shape)