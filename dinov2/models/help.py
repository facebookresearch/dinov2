import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Any, Callable, List, Optional, Tuple

class BaseModule(nn.Module):
    def __init__(self):
        super(BaseModule, self).__init__()

def conv1x1(in_planes: int, out_planes: int, stride: int = 1) -> nn.Conv2d:
    """1x1 convolution"""
    return nn.Conv2d(in_planes, out_planes, kernel_size=1, stride=stride, bias=False)

def conv3x3(in_channels, out_channels, stride=1, padding=1):
    """3x3 convolution with no bias."""
    return nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=stride, padding=padding, bias=False)

from torch import Tensor

class Merge_block(BaseModule):
    def __init__(self, fea_c, ada_c, mid_c, return_ada=True):
        super(Merge_block, self).__init__()

        self.fea_c = fea_c
        self.ada_c = ada_c
        # 784 - embedded dim + adapter_c
        self.embeded_dim = 768
        self.fc_1 = nn.Linear(self.embeded_dim*2, mid_c)
        print("Fc 1 type: ", self.fc_1.weight.dtype, self.fc_1.bias.dtype)
        self.fc_2 = nn.Linear(mid_c, self.embeded_dim)
        self.return_ada = return_ada

        if self.return_ada:
            self.conv_3 = nn.Conv1d(mid_c, self.embeded_dim, kernel_size=1)  # 1D Conv instead of 3x3
        else:
            self.conv_3 = None

    def forward(self, fea, adapter, ratio=1.0):
        res = fea
        # print("Before concatenation: ", fea.shape, adapter.shape, self.fea_c, self.ada_c)
        # print("before concatenation: ", fea.dtype, adapter.dtype)
        fea = torch.cat([fea, adapter], dim=-1)  # (B, seq_len, fea_c + ada_c)
        # print("after concatenation: ", fea.shape, adapter.shape)
        B, seq_len, C = fea.shape
        fea = fea.view(B * seq_len, C) 
        # print("before concatenation: ", fea.dtype, adapter.dtype)
        fea = fea.to(self.fc_1.weight.dtype)
        fea = self.fc_1(fea) 
        fea = fea.view(B, seq_len, -1)  
        ada = self.fc_2(fea)  
        fea_out = ratio * ada + res
        if self.return_ada:
           
            ada = self.conv_3(fea.permute(0, 2, 1))
            return fea_out, ada.permute(0, 2, 1)
        else:
            return fea_out, None



def conv7x7(in_planes: int, out_planes: int, stride: int = 3, groups: int = 1,  padding: int = 3, dilation: int = 1) -> nn.Conv2d:
    """7x7 convolution with padding"""
    return nn.Conv2d(in_planes, out_planes, kernel_size=7, stride=stride, padding=padding, groups=groups, bias=False, dilation=dilation)

class BasicBlock(nn.Module):
    expansion: int = 1

    def __init__(
        self,
        inplanes: int,
        planes: int,
        stride: int = 1,
        downsample: Optional[nn.Module] = None,
        groups: int = 1,
        base_width: int = 64,
        dilation: int = 1,
        norm_layer: Optional[Callable[..., nn.Module]] = None,
    ) -> None:
        super().__init__()
        if norm_layer is None:
            norm_layer = nn.LayerNorm
        if groups != 1 or base_width != 64:
            raise ValueError("BasicBlock only supports groups=1 and base_width=64")
        if dilation > 1:
            raise NotImplementedError("Dilation > 1 not supported in BasicBlock")
        
        # Both self.conv1 and self.downsample layers downsample the input when stride != 1
        self.conv1 = conv3x3(inplanes, planes, stride)
        self.bn1 = norm_layer([planes])  # Modify this to pass the correct shape
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = conv3x3(planes, planes)
        self.bn2 = norm_layer([planes])  # Modify this to pass the correct shape
        self.downsample = downsample
        self.stride = stride

    def forward(self, x: Tensor) -> Tensor:
        identity = x
        # x = x.to(self.expand_channels.weight.dtype)
        out = self.conv1(x)
        # Reshape for LayerNorm
        # C, H, W = out.shape
        if out.dim() == 3:
            out = out.unsqueeze(0)
        out = out.permute(0, 2, 3, 1)  # [B, C, H, W] -> [B, H, W, C]
        out = self.bn1(out)  # Apply LayerNorm on the channel dimension (last)
        out = out.permute(0, 3, 1, 2)  # [B, H, W, C] -> [B, C, H, W]
        out = self.relu(out)

        out = self.conv2(out)
        # if out.dim() == 3:
        #     out = out.unsqueeze(0)
        # Reshape for LayerNorm
        # C, H, W = out.shape
        out = out.permute(0, 2, 3, 1)  # [B, C, H, W] -> [B, H, W, C]
        out = self.bn2(out)  # Apply LayerNorm on the channel dimension (last)
        out = out.permute(0, 3, 1, 2)  # [B, H, W, C] -> [B, C, H, W]

        if self.downsample is not None:
            identity = self.downsample(x)

        out += identity
        out = self.relu(out)

        return out

class Model_level_Adapeter(BaseModule):
    def __init__(self, in_c=3, in_dim=8, w_lut=True):
        super(Model_level_Adapeter, self).__init__()
        self.conv_1 = conv3x3(in_c, in_c, 2)
        self.conv_2 = conv3x3(in_c, in_c, 2)
        self.conv_3 = conv3x3(in_c, in_c, 2)
        self.w_lut = w_lut
        if self.w_lut:  # With LUT: I1, I2, I3, I4
            self.conv_4 = conv3x3(in_c, in_c, 2)
            self.uni_conv = conv7x7(4*in_c, in_dim, 2, padding=3)
        else:  # Without LUT: I1, I2, I3
            self.uni_conv = conv7x7(3*in_c, in_dim, 2, padding=3)

        # self.res_1 = BasicBlock(inplanes=in_dim, planes=in_dim)
        # self.res_2 = BasicBlock(inplanes=in_dim, planes=in_dim)

    def forward(self, IMGS):
        if self.w_lut:
            adapter = torch.cat([self.conv_1(IMGS[0]), self.conv_2(IMGS[1]), self.conv_3(IMGS[2]), self.conv_4(IMGS[3])], dim=1)

        else:
            adapter = torch.cat([self.conv_1(IMGS[0]), self.conv_2(IMGS[1]), self.conv_3(IMGS[2])], dim=1)
        # print("Adapter:", adapter.shape)
        adapter = self.uni_conv(adapter)
        # print("Adapter:", adapter.shape)
        # adapter = self.res_1(adapter)
        # adapter = self.res_2(adapter)
        return adapter

# class Model_level_Adapeter(BaseModule):
#     def __init__(self, in_c=12, in_dim=16, w_lut=True):
#         super(Model_level_Adapeter, self).__init__()
#         self.conv_1 = conv3x3(in_c, in_c, 2)
#         self.conv_2 = conv3x3(in_c, in_c, 2)
#         self.conv_3 = conv3x3(in_c, in_c, 2)
#         self.w_lut = w_lut
#         if self.w_lut:
#             self.conv_4 = conv3x3(in_c, in_c, 2)
#             self.channel_reducer = conv7x7(272, in_dim, 2, padding=3)
#             self.uni_conv = conv7x7(12, in_dim, 2, padding=3)
#         else:
#             self.uni_conv = conv7x7(3*in_c, in_dim, 2, padding=3)

#         self.res_1 = BasicBlock(inplanes=in_dim, planes=in_dim)
#         self.res_2 = BasicBlock(inplanes=in_dim, planes=in_dim)
#         self.expand_channels = nn.Conv2d(3, 32, kernel_size=1, stride=1, padding=0).to(torch.float16)

#     def forward(self, IMGS):
#         device = IMGS[0].device 
#         print("DEVICE", device)
#         IMGS = [img.to(torch.float16) for img in IMGS]

#         reduce_channels = nn.Conv2d(32, 12, kernel_size=1, bias=False).to(device).to(torch.float16)

#         IMGS = [img.to(self.expand_channels.weight.dtype) for img in IMGS]
#         print("Types: ", IMGS[0].dtype, self.expand_channels.weight.dtype)
        
#         IMGS = [reduce_channels(self.expand_channels(img)) for img in IMGS]

#         print(f"After first reduction: {IMGS[0].shape}")
        

#         temp_conv_1 = nn.Conv2d(12, 12, kernel_size=3, stride=2, padding=1, bias=False).to(device).to(torch.float16)
#         temp_conv_2 = nn.Conv2d(12, 12, kernel_size=3, stride=2, padding=1, bias=False).to(device).to(torch.float16)
#         temp_conv_3 = nn.Conv2d(12, 12, kernel_size=3, stride=2, padding=1, bias=False).to(device).to(torch.float16)
#         print("HERE HERE")
#         if self.w_lut:
#             temp_conv_4 = nn.Conv2d(12, 12, kernel_size=3, stride=2, padding=1, bias=False).to(device).to(torch.float16)
#             print("HERE HERE 22", len(IMGS))
#             adapter = torch.cat([
#                 temp_conv_1(IMGS[0]), 
#                 temp_conv_2(IMGS[1]), 
#                 temp_conv_3(IMGS[2]), 
#                 temp_conv_4(IMGS[3])
#             ], dim=1)
#         else:
#             adapter = torch.cat([
#                 temp_conv_1(IMGS[0]), 
#                 temp_conv_2(IMGS[1]), 
#                 temp_conv_3(IMGS[2])
#             ], dim=1)
        
#         adapter = adapter.half()
#         print("HERE HERE 333")
#         adapter = self.uni_conv(adapter)
#         print("HERE HERE 44")
#         adapter = self.res_1(adapter)   # Residual Block 1 
#         adapter = self.res_2(adapter)   # Residual Block 2
#         return adapter

import torch
import torch.nn as nn
import torch.nn.functional as F

# class Input_level_Adapeter(nn.Module):
#     def __init__(self, mode='normal', lut_dim=32, k_size=3, w_lut=True, in_channels=3):
#         """
#         Args:
#             mode (str): Operating mode. Can be 'normal' or another mode if you extend this module.
#             lut_dim (int): The output channel dimension if using the LUT branch.
#             k_size (int): Kernel size for the convolutional layers.
#             w_lut (bool): Whether to use the LUT branch.
#             in_channels (int): Number of input channels. Typically 3 for RGB/RAW images.
#         """
#         super(Input_level_Adapeter, self).__init__()
#         self.mode = mode
#         self.lut_dim = lut_dim
#         self.k_size = k_size
#         self.w_lut = w_lut

#         # First convolutional block.
#         self.conv1 = nn.Conv2d(in_channels, 16, kernel_size=k_size, padding=k_size // 2, bias=False)
#         # self.bn1 = nn.LayerNorm(16)
#         self.relu = nn.ReLU(inplace=True)
        
#         # Second convolutional block.
#         self.conv2 = nn.Conv2d(16, 32, kernel_size=k_size, padding=k_size // 2, bias=False)
#         # self.bn2 = nn.LayerNorm(32)
#         self.bn1 = nn.GroupNorm(1, 16)
#         self.bn2 = nn.GroupNorm(1, 32)
#         # If using LUT processing, map the 32-channel features to lut_dim channels.
#         if self.w_lut:
#             self.lut_conv = nn.Conv2d(32, lut_dim, kernel_size=1, bias=False)
        
#         # Create two downsampling layers for multi-scale outputs.
#         self.down1 = nn.Conv2d(32 if not w_lut else lut_dim, 
#                                32 if not w_lut else lut_dim, 
#                                kernel_size=3, stride=2, padding=1, bias=False)
#         self.down2 = nn.Conv2d(32 if not w_lut else lut_dim, 
#                                32 if not w_lut else lut_dim, 
#                                kernel_size=3, stride=2, padding=1, bias=False)
        
#     def forward(self, x):
#         """
#         Forward pass for the input-level adapter.
#         Args:
#             x (Tensor): Input image tensor of shape (B, in_channels, H, W).
#         Returns:
#             List[Tensor]: A list of feature maps at multiple scales. For example:
#                           [feat_full, feat_down1, feat_down2]
#                           where feat_down2 is the most downsampled feature used for adaptation.
#         """
#         # Initial conv block.
#         out = self.conv1(x)
#         out = self.bn1(out)
#         out = self.relu(out)
        
#         out = self.conv2(out)
#         out = self.bn2(out)
#         out = self.relu(out)
        
#         # If enabled, adjust features via LUT branch.
#         if self.w_lut:
#             out = self.lut_conv(out)
        
#         # Compute multi-scale features.
#         feat_full = out                   # Original resolution feature.
#         feat_down1 = self.relu(self.down1(feat_full))  # Downsampled by a factor of 2.
#         feat_down2 = self.relu(self.down2(feat_down1))   # Downsampled further.
        
#         # Return a list of features. In your transformer, you can pick the desired scale.
#         return [feat_full, feat_down1, feat_down2]


class CustomLayerNorm(nn.Module):
    def __init__(self, normalized_shape):
        super().__init__()
        self.norm = nn.LayerNorm(normalized_shape)
        
    def forward(self, x):
        # Input has shape [B, S, D] or [B, C, H, W]
        if len(x.shape) == 3:  # [B, S, D] or [B, C, S]
            if x.shape[1] == self.norm.normalized_shape[0]:
                # This is [B, C, S] format, need to transpose
                x = x.transpose(1, 2)  # Now [B, S, C]
                x = self.norm(x)
                x = x.transpose(1, 2)  # Back to [B, C, S]
            else:
                # Already in [B, S, C] format
                x = self.norm(x)
        elif len(x.shape) == 4:  # [B, C, H, W]
            b, c, h, w = x.shape
            # Reshape to [B, H*W, C]
            x = x.permute(0, 2, 3, 1).reshape(b, h*w, c)
            # Apply norm
            x = self.norm(x)
            # Reshape back to [B, C, H, W]
            x = x.reshape(b, h, w, c).permute(0, 3, 1, 2)
        return x

# Predictor P_K
class Kernel_Predictor(nn.Module):
    def __init__(self, dim, mode='normal', num_heads=1, qkv_bias=True, qk_scale=None, attn_drop=0., proj_drop=0.):
        super().__init__()
        self.num_heads = num_heads
        head_dim = dim // num_heads
        # Use provided scale factor or default to head_dim^-0.5
        self.scale = qk_scale or head_dim ** -0.5
        # Query Adaptive Learning (QAL)
        self.q = nn.Parameter(torch.rand((1, 4, dim)), requires_grad=True)
        # self.input_proj = nn.Conv2d(16, 3, kernel_size=1) 
        self.kv_downsample = nn.Sequential(
            nn.Conv2d(3, dim // 8, kernel_size=3, stride=2, padding=1),
            nn.GroupNorm(1, dim // 8),  # replaced BatchNorm2d with GroupNorm
            nn.GELU(),
            nn.Conv2d(dim // 8, dim // 4, kernel_size=3, stride=2, padding=1),
            nn.GroupNorm(1, dim // 4),
            nn.GELU(),
            nn.Conv2d(dim // 4, dim // 2, kernel_size=3, stride=2, padding=1),
            nn.GroupNorm(1, dim // 2),
            nn.GELU(),
            nn.Conv2d(dim // 2, dim, kernel_size=3, stride=2, padding=1),
            nn.GroupNorm(1, dim),
        )

        self.k = nn.Linear(dim, dim, bias=qkv_bias)
        self.v = nn.Linear(dim, dim, bias=qkv_bias)
        self.attn_drop = nn.Dropout(attn_drop)
        self.proj = nn.Linear(dim, dim)
        self.proj_drop = nn.Dropout(proj_drop)
        self.down = nn.Linear(dim, 1)
        self.softmax = nn.Softmax(dim=2)
        self.relu = nn.ReLU()
        self.sigmoid = nn.Sigmoid()

        # Set basic parameters
        if mode == 'low':
            self.gain_base = nn.Parameter(torch.FloatTensor([3]), requires_grad=True)
        else:
            self.gain_base = nn.Parameter(torch.FloatTensor([1]), requires_grad=True)
        self.r1_base = nn.Parameter(torch.FloatTensor([3]), requires_grad=False)
        self.r2_base = nn.Parameter(torch.FloatTensor([2]), requires_grad=False)

    def forward(self, x):
        # print("X type: ", type(x), x.dim)
        # x = x[0]
        # x = self.input_proj(x)
        # output = self.kv_downsample(x)
        # print("Output type: ", type(output))  # This should print <class 'torch.Tensor'>, but it might print <class 'list'>

        d_x = self.kv_downsample(x).flatten(2).transpose(1, 2)  # [B, N, C]
        B, N, C = d_x.shape
        k = self.k(d_x).reshape(B, N, self.num_heads, C // self.num_heads).permute(0, 2, 1, 3)
        v = self.v(d_x).reshape(B, N, self.num_heads, C // self.num_heads).permute(0, 2, 1, 3)
        q = self.q.expand(B, -1, -1).view(B, -1, self.num_heads, C // self.num_heads).permute(0, 2, 1, 3)
        attn = (q @ k.transpose(-2, -1)) * self.scale
        attn = attn.softmax(dim=-1)
        attn = self.attn_drop(attn)
        out = (attn @ v).transpose(1, 2).reshape(B, 4, C)
        out = self.proj(out)
        out = self.proj_drop(out)
        out = self.down(out).squeeze(-1)
        out = torch.unbind(out, 1)
        r1, r2, gain, sigma = out[0], out[1], out[2], out[3]
        r1 = 0.1 * r1 + self.r1_base
        r2 = 0.1 * r2 + self.r2_base
        gain = gain + self.gain_base
        return r1, r2, gain, self.sigmoid(sigma)

        
class Matrix_Predictor(nn.Module):
    def __init__(self, dim, num_heads=1, qkv_bias=True, qk_scale=None, attn_drop=0., proj_drop=0.):
        super().__init__()
        self.num_heads = num_heads
        head_dim = dim // num_heads
        self.scale = qk_scale or head_dim ** -0.5
        # Query Adaptive Learning (QAL)
        self.q = nn.Parameter(torch.rand((1, 9 + 1, dim)), requires_grad=True)
        self.kv_downsample = nn.Sequential(
            nn.Conv2d(3, dim // 8, kernel_size=3, stride=2, padding=1),
            nn.GroupNorm(1, dim // 8),
            nn.GELU(),
            nn.Conv2d(dim // 8, dim // 4, kernel_size=3, stride=2, padding=1),
            nn.GroupNorm(1, dim // 4),
            nn.GELU(),
            nn.Conv2d(dim // 4, dim // 2, kernel_size=3, stride=2, padding=1),
            nn.GroupNorm(1, dim // 2),
            nn.GELU(),
            nn.Conv2d(dim // 2, dim, kernel_size=3, stride=2, padding=1),
            nn.GroupNorm(1, dim),
        )
        self.k = nn.Linear(dim, dim, bias=qkv_bias)
        self.v = nn.Linear(dim, dim, bias=qkv_bias)
        self.attn_drop = nn.Dropout(attn_drop)
        self.proj = nn.Linear(dim, dim)
        self.proj_drop = nn.Dropout(proj_drop)
        self.down = nn.Linear(dim, 1)
        self.softmax = nn.Softmax(dim=2)
        self.relu = nn.ReLU()
        self.ccm_base = nn.Parameter(torch.eye(3), requires_grad=False)

    def forward(self, x):
        d_x = self.kv_downsample(x).flatten(2).transpose(1, 2)
        B, N, C = d_x.shape
        k = self.k(d_x).reshape(B, N, self.num_heads, C // self.num_heads).permute(0, 2, 1, 3)
        v = self.v(d_x).reshape(B, N, self.num_heads, C // self.num_heads).permute(0, 2, 1, 3)
        q = self.q.expand(B, -1, -1).view(B, -1, self.num_heads, C // self.num_heads).permute(0, 2, 1, 3)
        attn = (q @ k.transpose(-2, -1)) * self.scale
        attn = attn.softmax(dim=-1)
        attn = self.attn_drop(attn)
        out = (attn @ v).transpose(1, 2).reshape(B, 9 + 1, C)
        out = self.proj(out)
        out = self.proj_drop(out)
        out = self.down(out)
        out, distance = out[:, :9, :], out[:, 9:, :].squeeze(-1)
        out = out.view(B, 3, 3)
        ccm_matrix = 0.1 * out + self.ccm_base
        distance = self.relu(distance) + 1
        return ccm_matrix, distance

# AAAI 2024 NILUT, we change the channel number to avoid much FLOPs
class NILUT(nn.Module):
    """
    Simple residual coordinate-based neural network for fitting 3D LUTs
    Official code: https://github.com/mv-lab/nilut
    """
    def __init__(self, in_features=3, hidden_features=32, hidden_layers=3, out_features=3, res=True):
        super().__init__()
        
        self.res = res
        self.net = []
        self.net.append(nn.Linear(in_features, hidden_features))
        self.net.append(nn.ReLU())
        
        for _ in range(hidden_layers):
            self.net.append(nn.Linear(hidden_features, hidden_features))
            self.net.append(nn.Tanh())
        
        self.net.append(nn.Linear(hidden_features, out_features))
        if not self.res:
            self.net.append(torch.nn.Sigmoid())
        
        self.net = nn.Sequential(*self.net)
    
    def forward(self, intensity):
        output = self.net(intensity)
        if self.res:
            output = output + intensity
            output = torch.clamp(output, 0.,1.)
        return output


def _is_tensor_a_torch_image(x: Tensor) -> bool:
    return x.ndim >= 2

def _assert_image_tensor(img: Tensor) -> None:
    if not _is_tensor_a_torch_image(img):
        raise TypeError("Tensor is not a torch image.")


def _get_gaussian_kernel1d(kernel_size: int, sigma: float) -> Tensor:
    ksize_half = (kernel_size - 1) * 0.5

    x = torch.linspace(-ksize_half, ksize_half, steps=kernel_size).to(sigma.device)
    #print(x.device)
    #print(sigma.device)
    pdf = torch.exp(-0.5 * (x / sigma).pow(2))
    kernel1d = pdf / pdf.sum()

    return kernel1d

def _get_gaussian_kernel2d(
    kernel_size: List[int], sigma: List[float], dtype: torch.dtype, device: torch.device
) -> Tensor:
    kernel1d_x = _get_gaussian_kernel1d(kernel_size[0], sigma[0]).to(device, dtype=dtype)
    kernel1d_y = _get_gaussian_kernel1d(kernel_size[1], sigma[1]).to(device, dtype=dtype)
    kernel2d = torch.mm(kernel1d_y[:, None], kernel1d_x[None, :])
    return kernel2d

def _cast_squeeze_in(img: Tensor, req_dtypes: List[torch.dtype]) -> Tuple[Tensor, bool, bool, torch.dtype]:
    need_squeeze = False
    # make image NCHW
    if img.ndim < 4:
        img = img.unsqueeze(dim=0)
        need_squeeze = True

    out_dtype = img.dtype
    need_cast = False
    if out_dtype not in req_dtypes:
        need_cast = True
        req_dtype = req_dtypes[0]
        img = img.to(req_dtype)
    return img, need_cast, need_squeeze, out_dtype

from torch.nn.functional import grid_sample, conv2d, interpolate, pad as torch_pad


def _cast_squeeze_out(img: Tensor, need_cast: bool, need_squeeze: bool, out_dtype: torch.dtype) -> Tensor:
    if need_squeeze:
        img = img.squeeze(dim=0)

    if need_cast:
        if out_dtype in (torch.uint8, torch.int8, torch.int16, torch.int32, torch.int64):
            # it is better to round before cast
            img = torch.round(img)
        img = img.to(out_dtype)

    return img

def gaussian_blur(img: Tensor, kernel_size: List[int], sigma: List[float]) -> Tensor:
    if not (isinstance(img, torch.Tensor)):
        raise TypeError(f"img should be Tensor. Got {type(img)}")

    _assert_image_tensor(img)

    dtype = img.dtype if torch.is_floating_point(img) else torch.float32
    kernel = _get_gaussian_kernel2d(kernel_size, sigma, dtype=dtype, device=img.device)
    kernel = kernel.expand(img.shape[-3], 1, kernel.shape[0], kernel.shape[1])

    img, need_cast, need_squeeze, out_dtype = _cast_squeeze_in(
        img,
        [
            kernel.dtype,
        ],
    )

    # padding = (left, right, top, bottom)
    padding = [kernel_size[0] // 2, kernel_size[0] // 2, kernel_size[1] // 2, kernel_size[1] // 2]
    img = torch_pad(img, padding, mode="reflect")
    img = conv2d(img, kernel, groups=img.shape[-3])

    img = _cast_squeeze_out(img, need_cast, need_squeeze, out_dtype)
    return img


def Gain_Denoise(I1, r1, r2, gain, sigma, k_size=1):  # [9, 9] in LOD dataset, [3, 3] in other dataset
    out = []
    for i in range(I1.shape[0]):
        I1_gain = gain[i] * I1[i,:,:,:]
        blur = gaussian_blur(I1_gain, \
                                [k_size, k_size], \
                                [r1[i], r2[i]])
        sharp = blur + sigma[i] * (I1[i,:,:,:] - blur)
        out.append(sharp)
    return torch.stack([out[i] for i in range(I1.shape[0])], dim=0)


def SoG_algo(img, p=1):
    # https://library.imaging.org/admin/apis/public/api/ist/website/downloadArticle/cic/12/1/art00008
    img = img.permute(1,2,0)       # (C,H,W) --> (H,W,C)

    img_P = torch.pow(img, p)
    
    R_avg = torch.mean(img_P[:,:,0]) ** (1/p)
    G_avg = torch.mean(img_P[:,:,1]) ** (1/p)
    B_avg = torch.mean(img_P[:,:,2]) ** (1/p)

    Avg = torch.mean(img_P) ** (1/p)

    R_avg = R_avg / Avg
    G_avg = G_avg / Avg
    B_avg = B_avg / Avg

    img_out = torch.stack([img[:,:,0]/R_avg, img[:,:,1]/G_avg, img[:,:,2]/B_avg], dim=-1)

    return img_out 

def WB_CCM(I2, ccm_matrix, distance):
    out_I3 = []
    out_I4 = []
    for i in range(I2.shape[0]):
        # SOG White Balance Algorithm
        I3 = SoG_algo(I2[i,:,:,:])
        
        # Camera Color Matrix
        I4 = torch.tensordot(I3, ccm_matrix[i,:,:], dims=[[-1], [-1]])
        I4 = torch.clamp(I4, 1e-7, 1.0)
         
        out_I3.append(I3)
        out_I4.append(I4)

    return  torch.stack([out_I3[i] for i in range(I2.shape[0])], dim=0), \
            torch.stack([out_I4[i] for i in range(I2.shape[0])], dim=0)

class VitInputLevelAdapter(nn.Module):
    def __init__(self, mode='normal', lut_dim=32, out='all', k_size=3, w_lut=True):
        """
        Args:
            mode (str): Operating mode, e.g. 'normal' for normal/over-exposure or 'low' for low-light.
            lut_dim (int): Dimensionality for the implicit neural LUT.
            k_size (int): Kernel size for the denoising operation.
            w_lut (bool): Whether to use the implicit 3D Look-Up Table.
        """
        super(VitInputLevelAdapter, self).__init__()
        # These submodules predict transformation parameters from the input image.
        self.Predictor_K = Kernel_Predictor(dim=64, mode=mode)
        self.Predictor_M = Matrix_Predictor(dim=64)
        self.w_lut = w_lut
        if self.w_lut:
            self.LUT = NILUT(hidden_features=lut_dim)    
            print("hidden_features", lut_dim)
            # self.LUT = nn.Linear(224, 32)
        self.k_size = k_size
        self.out = out

    def forward(self, I1):
        # (1). I1 --> I2: Denoise & Enhancement & Sharpen
        r1, r2, gain, sigma = self.Predictor_K(I1)
        I2 = Gain_Denoise(I1, r1, r2, gain, sigma, k_size=self.k_size)  # (B,C,H,W)
        I2 = torch.clamp(I2, 1e-7, 1.0) # normal & over-exposure
        
        ccm_matrix, distance = self.Predictor_M(I2)
        # (2). I2 --> I3: White Balance, Shade of Gray
        # (3). I3 --> I4: Camera Colour Matrix Transformation
        I3, I4 = WB_CCM(I2, ccm_matrix, distance) # (B,H,W,C)
        
        if self.w_lut:
        # (4). I4 --> I5: Implicit Neural LUT
            I5 = self.LUT(I4).permute(0,3,1,2)
            
            if self.out == 'all':   # return all features
                return [I1, I2, I3.permute(0,3,1,2), I4.permute(0,3,1,2), I5]
            else:   # only return I5
                return [I5]
        
        else:
            if self.out == 'all':
                return [I1, I2, I3.permute(0,3,1,2), I4.permute(0,3,1,2)]
            else:
                return [I4.permute(0,3,1,2)]