import h5py
import torch
import random

import torch.nn as nn
import torch.nn.functional as F

from PIL import ImageFilter, Image
from torchvision import transforms, models


########## HEADER ##########
header = '''    
     _______               _______ _______   ________________    _______         _       _      __________       _______ 
|\     /(  ___  )     |\     /(  ___  |  ____ \  \__   __(  ____ \  (  ____ )\     /( (    /( (    /\__   __( (    /(  ____ \\
| )   ( | (   ) |     | )   ( | (   ) | (    \/     ) (  | (    \/  | (    )| )   ( |  \  ( |  \  ( |  ) (  |  \  ( | (    \/
| |   | | |   | |_____| |   | | (___) | (__         | |  | (_____   | (____)| |   | |   \ | |   \ | |  | |  |   \ | | |      
( (   ) ) |   | (_____| (   ) )  ___  |  __)        | |  (_____  )  |     __) |   | | (\ \) | (\ \) |  | |  | (\ \) | | ____ 
 \ \_/ /| | /\| |      \ \_/ /| (   ) | (           | |        ) |  | (\ (  | |   | | | \   | | \   |  | |  | | \   | | \_  )
  \   / | (_\ \ |       \   / | )   ( | (____/\  ___) (__/\____) |  | ) \ \_| (___) | )  \  | )  \  |__) (__| )  \  | (___) |
   \_/  (____\/_)        \_/  |/     \(_______/  \_______|_______)  |/   \__(_______)/    )_)/    )_)_______//    )_|_______)    
'''


########## DATA AUGMENTATIONS ##########
class GaussianBlur(object):
    """
    Apply Gaussian Blur to the PIL image.
    """
    def __init__(self, p=0.5, radius_min=0.1, radius_max=2.):
        self.prob = p
        self.radius_min = radius_min
        self.radius_max = radius_max

    def __call__(self, img):
        do_it = random.random() <= self.prob
        if not do_it:
            return img

        return img.filter(
            ImageFilter.GaussianBlur(
                radius=random.uniform(self.radius_min, self.radius_max)
            )
        )


class ContrastBrightnessJitter(object):
    """
    Adjust the contrast and brightness of a tensor/image
    """
    def __init__(self, p, alpha_min=0.5, alpha_max=1.5, beta_min=-0.25, beta_max=0.25) -> None:
        self.p = p
        self.alpha_min = alpha_min # min contrast
        self.beta_min = beta_min # min brightness
        self.alpha_max = alpha_max # max contrast
        self.beta_max = beta_max # max brightness

    def __call__(self, img):
        if random.random() < self.p:
            return contrast_brightness_jitter(img, self.alpha_min, self.alpha_max, self.beta_min, self.beta_max)
        else:
            return img


def contrast_brightness_jitter(img: torch.Tensor, alpha_min: float, alpha_max:float, beta_min: float, beta_max: float) -> torch.Tensor:
    alpha = random.uniform(alpha_min, alpha_max)
    beta = random.uniform(beta_min, beta_max)
    return torch.clip(alpha * img + beta, 0.0, 1.0)
    

########## NETWORKS ##########
class Residual(nn.Module):
    '''
    A class defining the residual block of ResNet.

    Args:
        in_channels: number of channels in the input image.
        num_hiddens: number of channels produced by the residual block.
        num_residual_hiddens: number of channels produced by the convolution.
    '''
    def __init__(self, in_channels, num_hiddens, num_residual_hiddens):
        super(Residual, self).__init__()
        self._block = nn.Sequential(
            nn.ReLU(True),
            nn.Conv2d(in_channels=in_channels,
                      out_channels=num_residual_hiddens,
                      kernel_size=3, stride=1, padding=1, bias=False),
            nn.ReLU(True),
            nn.Conv2d(in_channels=num_residual_hiddens,
                      out_channels=num_hiddens,
                      kernel_size=1, stride=1, bias=False)
        )
    
    def forward(self, x):
        return x + self._block(x)


class ResidualStack(nn.Module):
    '''
    A class defining stacked residual blocks of ResNet.

    Args:
        in_channels: number of channels in the input image.
        num_hiddens: number of channels produced by the residual block.
        num_residual_layers: number of stacked residual blocks.
        num_residual_hiddens: number of channels produced by the convolution.
    '''
    def __init__(self, in_channels, num_hiddens, num_residual_layers, num_residual_hiddens):
        super(ResidualStack, self).__init__()
        self._num_residual_layers = num_residual_layers
        self._layers = nn.ModuleList([Residual(in_channels, num_hiddens, num_residual_hiddens) for _ in range(self._num_residual_layers)])

    def forward(self, x):
        for i in range(self._num_residual_layers):
            x = self._layers[i](x)
        return F.relu(x)


def CustomVGG16():
    model = models.vgg16(pretrained=True)
    
    pre_conv_layer = [nn.Conv2d(in_channels=15,
                                out_channels=3,
                                kernel_size=3,
                                stride=1,
                                padding=1)]
    
    pre_conv_layer.extend(list(model.features)[:10])
    
    model = nn.Sequential(*pre_conv_layer)

    # for param in model.features[1:].parameters():
    #         param.requires_grad = False
    
    return model