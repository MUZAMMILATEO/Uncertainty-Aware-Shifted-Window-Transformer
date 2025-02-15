'''
UAST-Net model

Swin-Transformer code retrieved from:
https://github.com/SwinTransformer/Swin-Transformer-Semantic-Segmentation

Original paper:
Liu, Z., Lin, Y., Cao, Y., Hu, H., Wei, Y., Zhang, Z., ... & Guo, B. (2021).
Swin transformer: Hierarchical vision transformer using shifted windows.
arXiv preprint arXiv:2103.14030.

Modified and tested by:
Junyu Chen
jchen245@jhmi.edu
Johns Hopkins University
'''

import torch
import torch.nn as nn
import torch.utils.checkpoint as checkpoint
from timm.models.layers import DropPath, trunc_normal_, to_3tuple, to_2tuple
from torch.distributions.normal import Normal
import torch.nn.functional as nnf
import numpy as np
import models.configs_UAST_Net as configs
from torch.nn.functional import leaky_relu
from torch.nn import init




class Mlp(nn.Module):
    def __init__(self, in_features, hidden_features=None, out_features=None, act_layer=nn.GELU, drop=0.15):
        super().__init__()
        out_features = out_features or in_features
        hidden_features = hidden_features or in_features
        self.fc1 = nn.Linear(in_features, hidden_features)
        self.act = act_layer()
        self.fc2 = nn.Linear(hidden_features, out_features)
        self.drop = drop

    def forward(self, x):
        x = self.fc1(x)
        x = self.act(x)
        x = nnf.dropout(x, self.drop, training=self.training)
        x = self.fc2(x)
        x = nnf.dropout(x, self.drop, training=self.training)
        return x


def window_partition(x, window_size):
    """
    Args:
        x: (B, H, W, C)
        window_size (int): window size
    Returns:
        windows: (num_windows*B, window_size, window_size, C)
    """
    B, H, W, C = x.shape
    x = x.view(B, H // window_size[0], window_size[0], W // window_size[1], window_size[1], C)
    windows = x.permute(0, 1, 3, 2, 4, 5).contiguous().view(-1, window_size[0], window_size[1], C)         # Breaking the entire image into a batch of windows (not patches)
    return windows

def window_reverse(windows, window_size, H, W):
    """
    Args:
        windows: (num_windows*B, window_size, window_size, C)
        window_size (int): Window size
        H (int): Height of image
        W (int): Width of image
    Returns:
        x: (B, H, W, C)
    """
    B = int(windows.shape[0] / (H * W / window_size[0] / window_size[1]))
    x = windows.view(B, H // window_size[0], W // window_size[1], window_size[0], window_size[1], -1)
    x = x.permute(0, 1, 3, 2, 4, 5).contiguous().view(B, H, W, -1)
    return x

class WindowAttention(nn.Module):
    def __init__(self, dim, window_size, num_heads, qkv_bias=True, qk_scale=None, rpe=True, attn_drop=0.15, proj_drop=0.15):
        super().__init__()
        self.dim = dim                               # patch embedding dimension
        self.window_size = window_size  # Wh, Ww     # patch height and width
        self.num_heads = num_heads                   
        head_dim = dim // num_heads                 # head dim = embedding space dimension / number of heads
        self.scale = qk_scale or head_dim ** -0.5    # if head dim is Dk, then softmax(((QK^t)/sqrt(Dk))*V), the factor controls the over-shooting

        self.relative_position_bias_table = nn.Parameter(
            torch.zeros((2 * window_size[0] - 1) * (2 * window_size[1] - 1), num_heads))  # 2*Wh-1 * 2*Ww-1, nH

        coords_h = torch.arange(self.window_size[0])
        coords_w = torch.arange(self.window_size[1])
        coords = torch.stack(torch.meshgrid([coords_h, coords_w]))  # 2, Wh, Ww
        coords_flatten = torch.flatten(coords, 1)  # 2, Wh*Ww
        self.rpe = rpe
        if self.rpe:
            relative_coords = coords_flatten[:, :, None] - coords_flatten[:, None, :]  # 2, Wh*Ww, Wh*Ww
            relative_coords = relative_coords.permute(1, 2, 0).contiguous()  # Wh*Ww, Wh*Ww, 2
            relative_coords[:, :, 0] += self.window_size[0] - 1  # shift to start from 0
            relative_coords[:, :, 1] += self.window_size[1] - 1
            relative_coords[:, :, 0] *= 2 * self.window_size[1] - 1
            relative_position_index = relative_coords.sum(-1)  # Wh*Ww, Wh*Ww
            self.register_buffer("relative_position_index", relative_position_index)

        self.qkv = nn.Linear(dim, dim * 3, bias=qkv_bias)
        self.attn_drop = nn.Dropout(attn_drop)
        self.proj = nn.Linear(dim, dim)
        self.proj_drop = nn.Dropout(proj_drop)

        trunc_normal_(self.relative_position_bias_table, std=.02)
        self.softmax = nn.Softmax(dim=-1)

    def forward(self, x, mask=None):
        B_, N, C = x.shape                      # B_, N, C are the batch dimension, number of patches, and patch embedding space dimension, respectively
        qkv = self.qkv(x).reshape(B_, N, 3, self.num_heads, C // self.num_heads).permute(2, 0, 3, 1, 4)
        q, k, v = qkv[0], qkv[1], qkv[2]               

        q = q * self.scale
        attn = (q @ k.transpose(-2, -1))

        if self.rpe:
            relative_position_bias = self.relative_position_bias_table[self.relative_position_index.view(-1)].view(
                self.window_size[0] * self.window_size[1], self.window_size[0] * self.window_size[1], -1)  # Wh*Ww,Wh*Ww,nH
            relative_position_bias = relative_position_bias.permute(2, 0, 1).contiguous()  # nH, Wh*Ww, Wh*Ww
            attn = attn + relative_position_bias.unsqueeze(0)

        if mask is not None:
            nW = mask.shape[0]
            attn = attn.view(B_ // nW, nW, self.num_heads, N, N) + mask.unsqueeze(1).unsqueeze(0)
            attn = attn.view(-1, self.num_heads, N, N)
            attn = self.softmax(attn)
        else:
            attn = self.softmax(attn)

        attn = nnf.dropout(attn, self.attn_drop.p, training=self.training)
        x = (attn @ v).transpose(1, 2).reshape(B_, N, C)
        x = self.proj(x)
        x = nnf.dropout(x, self.proj_drop.p, training=self.training)
        return x


class SwinTransformerBlock(nn.Module):
    def __init__(self, dim, num_heads, window_size=(7, 7), shift_size=(0, 0), mlp_ratio=4., qkv_bias=True, qk_scale=None, rpe=True, drop=0., attn_drop=0.15, drop_path=0., act_layer=nn.GELU, norm_layer=nn.LayerNorm, MC_drop=0.15):
        super().__init__()
        self.dim = dim
        self.num_heads = num_heads
        self.window_size = window_size
        self.shift_size = shift_size
        self.mlp_ratio = mlp_ratio
        assert 0 <= min(self.shift_size) < min(self.window_size), "shift_size must in 0-window_size, shift_sz: {}, win_size: {}".format(self.shift_size, self.window_size)

        self.norm1 = norm_layer(dim)
        self.attn = WindowAttention(
            dim, window_size=self.window_size, num_heads=num_heads,
            qkv_bias=qkv_bias, qk_scale=qk_scale, rpe=rpe, attn_drop=attn_drop, proj_drop=drop) # Output has the same dimensions (B, N, C) as that of the input

        self.drop_path = DropPath(drop_path) if drop_path > 0. else nn.Identity()
        self.norm2 = norm_layer(dim)
        mlp_hidden_dim = int(dim * mlp_ratio)
        self.mlp = Mlp(in_features=dim, hidden_features=mlp_hidden_dim, act_layer=act_layer, drop=MC_drop)

        self.H = None
        self.W = None

    def forward(self, x, mask_matrix):
        B, L, C = x.shape                           # B, L, C represent the batch size, the total number of patch embeddings, and the dimension of the embedding space, repsectively
        H, W = self.H, self.W
        assert L == H * W, "input feature has wrong size"

        shortcut = x
        x = self.norm1(x)
        x = x.view(B, H, W, C)

        # pad feature maps to multiples of window size
        pad_l = pad_t = 0
        pad_r = (self.window_size[0] - H % self.window_size[0]) % self.window_size[0]
        pad_b = (self.window_size[1] - W % self.window_size[1]) % self.window_size[1]
        x = nnf.pad(x, (0, 0, pad_l, pad_r, pad_t, pad_b))
        _, Hp, Wp, _ = x.shape                                               # Hp and Wp are the new height and width of the input image

        # cyclic shift
        if min(self.shift_size) > 0:
            shifted_x = torch.roll(x, shifts=(-self.shift_size[0], -self.shift_size[1]), dims=(1, 2))
            attn_mask = mask_matrix
        else:
            shifted_x = x
            attn_mask = None

        # partition windows
        x_windows = window_partition(shifted_x, self.window_size)  # nW*B, window_size, window_size, C
        x_windows = x_windows.view(-1, self.window_size[0] * self.window_size[1], C)  # nW*B, window_size*window_size, C       This step is required since the shape of the input to attention mechanism is (B, N, C)

        # W-MSA/SW-MSA
        attn_windows = self.attn(x_windows, mask=attn_mask)  # nW*B, window_size*window_size, C

        # merge windows
        attn_windows = attn_windows.view(-1, self.window_size[0], self.window_size[1], C)
        shifted_x = window_reverse(attn_windows, self.window_size, Hp, Wp)  # B H' W' C

        # reverse cyclic shift
        if min(self.shift_size) > 0:
            x = torch.roll(shifted_x, shifts=(self.shift_size[0], self.shift_size[1]), dims=(1, 2))
        else:
            x = shifted_x

        if pad_r > 0 or pad_b > 0:
            x = x[:, :H, :W, :].contiguous()

        x = x.view(B, H * W, C)                          # The x is organized for each batch sample as a list of patches where H*W is the total number of patches and C their embedding dimension

        # FFN
        x = shortcut + self.drop_path(x)
        x = x + self.drop_path(self.mlp(self.norm2(x)))

        return x

class PatchMerging(nn.Module):
    """ Patch Merging Layer
    Args:
        dim (int): Number of input channels.
        norm_layer (nn.Module, optional): Normalization layer.  Default: nn.LayerNorm
    """
    def __init__(self, dim, norm_layer=nn.LayerNorm):
        super().__init__()
        self.dim = dim
        self.reduction = nn.Linear(4 * dim, 2 * dim, bias=False)
        self.norm = norm_layer(4 * dim)

    def forward(self, x, H, W):
        """ Forward function.
        Args:
            x: Input feature, tensor size (B, H*W, C).
            H, W: Spatial resolution of the input feature.
        """
        B, L, C = x.shape
        assert L == H * W, "input feature has wrong size"              # H and W denote the number of patches (not pixels) along rows and columns

        x = x.view(B, H, W, C)

        # padding
        pad_input = (H % 2 == 1) or (W % 2 == 1)
        if pad_input:
            x = nnf.pad(x, (0, 0, 0, W % 2, 0, H % 2))

        x0 = x[:, 0::2, 0::2, :]  # B H/2 W/2 C
        x1 = x[:, 1::2, 0::2, :]  # B H/2 W/2 C
        x2 = x[:, 0::2, 1::2, :]  # B H/2 W/2 C
        x3 = x[:, 1::2, 1::2, :]  # B H/2 W/2 C
        x = torch.cat([x0, x1, x2, x3], -1)  # B H/2 W/2 4*C
        x = x.view(B, -1, 4 * C)  # B H/2*W/2 4*C

        x = self.norm(x)
        x = self.reduction(x)

        return x

class BasicLayer(nn.Module):
    """ A basic Swin Transformer layer for one stage.
    Args:
        dim (int): Number of feature channels
        depth (int): Depths of this stage.
        num_heads (int): Number of attention head.
        window_size (int): Local window size. Default: 7.
        mlp_ratio (float): Ratio of mlp hidden dim to embedding dim. Default: 4.
        qkv_bias (bool, optional): If True, add a learnable bias to query, key, value. Default: True
        qk_scale (float | None, optional): Override default qk scale of head_dim ** -0.5 if set.
        drop (float, optional): Dropout rate. Default: 0.0
        attn_drop (float, optional): Attention dropout rate. Default: 0.0
        drop_path (float | tuple[float], optional): Stochastic depth rate. Default: 0.0
        norm_layer (nn.Module, optional): Normalization layer. Default: nn.LayerNorm
        downsample (nn.Module | None, optional): Downsample layer at the end of the layer. Default: None
        use_checkpoint (bool): Whether to use checkpointing to save memory. Default: False.
    """

    def __init__(self,
                 dim,
                 depth,
                 num_heads,
                 window_size=(7, 7, 7),
                 mlp_ratio=4.,
                 qkv_bias=True,
                 qk_scale=None,
                 rpe=True,
                 drop=0.,
                 attn_drop=0.,
                 drop_path=0.,
                 norm_layer=nn.LayerNorm,
                 downsample=None,
                 use_checkpoint=False,
                 pat_merg_rf=2,
                 MC_drop=0.15):
        super().__init__()
        self.window_size = window_size
        self.shift_size = (window_size[0] // 2, window_size[1] // 2)
        self.depth = depth
        self.use_checkpoint = use_checkpoint
        self.pat_merg_rf = pat_merg_rf
        # build blocks
        self.blocks = nn.ModuleList([
            SwinTransformerBlock(
                dim=dim,
                num_heads=num_heads,
                window_size=window_size,
                shift_size=(0, 0) if (i % 2 == 0) else (window_size[0] // 2, window_size[1] // 2),
                mlp_ratio=mlp_ratio,
                qkv_bias=qkv_bias,
                qk_scale=qk_scale,
                rpe=rpe,
                drop=drop,
                attn_drop=attn_drop,
                drop_path=drop_path[i] if isinstance(drop_path, list) else drop_path,
                norm_layer=norm_layer,
                MC_drop=MC_drop)
            for i in range(depth)])

        # patch merging layer
        if downsample is not None:
            self.downsample = downsample(dim=dim, norm_layer=norm_layer)
        else:
            self.downsample = None

    def forward(self, x, H, W):
        """ Forward function.
        Args:
            x: Input feature, tensor size (B, H*W, C).
            H, W: Spatial resolution of the input feature.
        """

        # calculate attention mask for SW-MSA
        Hp = int(np.ceil(H / self.window_size[0])) * self.window_size[0]
        Wp = int(np.ceil(W / self.window_size[1])) * self.window_size[1]
        img_mask = torch.zeros((1, Hp, Wp, 1), device=x.device)  # 1 Hp Wp 1
        h_slices = (slice(0, -self.window_size[0]),
                    slice(-self.window_size[0], -self.shift_size[0]),
                    slice(-self.shift_size[0], None))
        w_slices = (slice(0, -self.window_size[1]),
                    slice(-self.window_size[1], -self.shift_size[1]),
                    slice(-self.shift_size[1], None))
        cnt = 0
        for h in h_slices:
            for w in w_slices:
                img_mask[:, h, w, :] = cnt
                cnt += 1

        mask_windows = window_partition(img_mask, self.window_size)  # nW, window_size, window_size, 1
        mask_windows = mask_windows.view(-1, self.window_size[0] * self.window_size[1])
        attn_mask = mask_windows.unsqueeze(1) - mask_windows.unsqueeze(2)
        attn_mask = attn_mask.masked_fill(attn_mask != 0, float(-100.0)).masked_fill(attn_mask == 0, float(0.0))

        for blk in self.blocks:
            blk.H, blk.W = H, W
            if self.use_checkpoint:
                x = checkpoint.checkpoint(blk, x, attn_mask)
            else:
                x = blk(x, attn_mask)
        if self.downsample is not None:
            x_down = self.downsample(x, H, W)
            Wh, Ww = (H + 1) // 2, (W + 1) // 2
            return x, H, W, x_down, Wh, Ww
        else:
            return x, H, W, x, H, W


class PatchEmbed(nn.Module):
    """ Image to Patch Embedding
    Args:
        patch_size (int): Patch token size. Default: 4.
        in_chans (int): Number of input image channels. Default: 3.
        embed_dim (int): Number of linear projection output channels. Default: 96.
        norm_layer (nn.Module, optional): Normalization layer. Default: None
    """

    def __init__(self, patch_size=4, in_chans=3, embed_dim=96, norm_layer=None):
        super().__init__()
        patch_size = to_2tuple(patch_size)
        self.patch_size = patch_size

        self.in_chans = in_chans
        self.embed_dim = embed_dim

        self.proj = nn.Conv2d(in_chans, embed_dim, kernel_size=patch_size, stride=patch_size)
        if norm_layer is not None:
            self.norm = norm_layer(embed_dim)
        else:
            self.norm = None

    def forward(self, x):
        """Forward function."""
        # padding
        _, _, H, W = x.size()                            # The input is a raw RGB image with height H and width W
        if W % self.patch_size[1] != 0:
            x = nnf.pad(x, (0, self.patch_size[1] - W % self.patch_size[1]))
        if H % self.patch_size[0] != 0:
            x = nnf.pad(x, (0, 0, 0, self.patch_size[0] - H % self.patch_size[0]))

        x = self.proj(x)  # B C Wh Ww                  # Here Wh and Ww represent the bumber of patches along rows and columns, while C is the embedding dimension
        if self.norm is not None:
            Wh, Ww = x.size(2), x.size(3)
            x = x.flatten(2).transpose(1, 2)
            x = self.norm(x)
            x = x.transpose(1, 2).view(-1, self.embed_dim, Wh, Ww)

        return x

class SinusoidalPositionEmbedding(nn.Module):
    '''
    Rotary Position Embedding
    '''
    def __init__(self,):
        super(SinusoidalPositionEmbedding, self).__init__()

    def forward(self, x):
        batch_sz, n_patches, hidden = x.shape
        position_ids = torch.arange(0, n_patches).float().cuda()
        indices = torch.arange(0, hidden//2).float().cuda()
        indices = torch.pow(10000.0, -2 * indices / hidden)
        embeddings = torch.einsum('b,d->bd', position_ids, indices)
        embeddings = torch.stack([torch.sin(embeddings), torch.cos(embeddings)], dim=-1)
        embeddings = torch.reshape(embeddings, (1, n_patches, hidden))
        return embeddings

class SinPositionalEncoding3D(nn.Module):
    def __init__(self, channels):
        """
        :param channels: The last dimension of the tensor you want to apply pos emb to.
        """
        super(SinPositionalEncoding3D, self).__init__()
        channels = int(np.ceil(channels/6)*2)
        if channels % 2:
            channels += 1
        self.channels = channels
        self.inv_freq = 1. / (10000 ** (torch.arange(0, channels, 2).float() / channels))
        #self.register_buffer('inv_freq', inv_freq)

    def forward(self, tensor):
        """
        :param tensor: A 5d tensor of size (batch_size, x, y, z, ch)
        :return: Positional Encoding Matrix of size (batch_size, x, y, z, ch)
        """
        tensor = tensor.permute(0, 2, 3, 4, 1)
        if len(tensor.shape) != 5:
            raise RuntimeError("The input tensor has to be 5d!")
        batch_size, x, y, z, orig_ch = tensor.shape
        pos_x = torch.arange(x, device=tensor.device).type(self.inv_freq.type())
        pos_y = torch.arange(y, device=tensor.device).type(self.inv_freq.type())
        pos_z = torch.arange(z, device=tensor.device).type(self.inv_freq.type())
        sin_inp_x = torch.einsum("i,j->ij", pos_x, self.inv_freq)
        sin_inp_y = torch.einsum("i,j->ij", pos_y, self.inv_freq)
        sin_inp_z = torch.einsum("i,j->ij", pos_z, self.inv_freq)
        emb_x = torch.cat((sin_inp_x.sin(), sin_inp_x.cos()), dim=-1).unsqueeze(1).unsqueeze(1)
        emb_y = torch.cat((sin_inp_y.sin(), sin_inp_y.cos()), dim=-1).unsqueeze(1)
        emb_z = torch.cat((sin_inp_z.sin(), sin_inp_z.cos()), dim=-1)
        emb = torch.zeros((x,y,z,self.channels*3),device=tensor.device).type(tensor.type())
        emb[:,:,:,:self.channels] = emb_x
        emb[:,:,:,self.channels:2*self.channels] = emb_y
        emb[:,:,:,2*self.channels:] = emb_z
        emb = emb[None,:,:,:,:orig_ch].repeat(batch_size, 1, 1, 1, 1)
        return emb.permute(0, 4, 1, 2, 3)

class SwinTransformer(nn.Module):
    r""" Swin Transformer
        A PyTorch impl of : `Swin Transformer: Hierarchical Vision Transformer using Shifted Windows`  -
          https://arxiv.org/pdf/2103.14030
    Args:
        img_size (int | tuple(int)): Input image size. Default 224
        patch_size (int | tuple(int)): Patch size. Default: 4
        in_chans (int): Number of input image channels. Default: 3
        num_classes (int): Number of classes for classification head. Default: 1000
        embed_dim (int): Patch embedding dimension. Default: 96
        depths (tuple(int)): Depth of each Swin Transformer layer.
        num_heads (tuple(int)): Number of attention heads in different layers.
        window_size (tuple): Window size. Default: 7
        mlp_ratio (float): Ratio of mlp hidden dim to embedding dim. Default: 4
        qkv_bias (bool): If True, add a learnable bias to query, key, value. Default: True
        qk_scale (float): Override default qk scale of head_dim ** -0.5 if set. Default: None
        drop_rate (float): Dropout rate. Default: 0
        attn_drop_rate (float): Attention dropout rate. Default: 0
        drop_path_rate (float): Stochastic depth rate. Default: 0.1
        norm_layer (nn.Module): Normalization layer. Default: nn.LayerNorm.
        ape (bool): If True, add absolute position embedding to the patch embedding. Default: False
        patch_norm (bool): If True, add normalization after patch embedding. Default: True
        use_checkpoint (bool): Whether to use checkpointing to save memory. Default: False
    """

    def __init__(self, pretrain_img_size=224,
                 patch_size=4,
                 in_chans=3,
                 embed_dim=96,
                 depths=[2, 2, 6, 2],
                 num_heads=[3, 6, 12, 24],
                 window_size=(7, 7, 7),
                 mlp_ratio=4.,
                 qkv_bias=True,
                 qk_scale=None,
                 drop_rate=0.,
                 attn_drop_rate=0.,
                 drop_path_rate=0.2,
                 norm_layer=nn.LayerNorm,
                 ape=False,
                 spe=False,
                 rpe=True,
                 patch_norm=True,
                 out_indices=(0, 1, 2, 3),
                 frozen_stages=-1,
                 use_checkpoint=False,
                 pat_merg_rf=2,
                 MC_drop=0.15):
        super().__init__()
        self.pretrain_img_size = pretrain_img_size
        self.num_layers = len(depths)
        self.embed_dim = embed_dim
        self.ape = ape
        self.spe = spe
        self.rpe = rpe
        self.patch_norm = patch_norm
        self.out_indices = out_indices
        self.frozen_stages = frozen_stages
        self.MC_drop = MC_drop
        # split image into non-overlapping patches
        self.patch_embed = PatchEmbed(
            patch_size=patch_size, in_chans=in_chans, embed_dim=embed_dim,
            norm_layer=norm_layer if self.patch_norm else None)

        # absolute position embedding
        if self.ape:
            pretrain_img_size = to_2tuple(pretrain_img_size)
            patch_size = to_2tuple(patch_size)
            patches_resolution = [pretrain_img_size[0] // patch_size[0], pretrain_img_size[1] // patch_size[1]]

            self.absolute_pos_embed = nn.Parameter(
                torch.zeros(1, embed_dim, patches_resolution[0], patches_resolution[1]))
            trunc_normal_(self.absolute_pos_embed, std=.02)
        elif self.spe:
            # self.pos_embd = SinPositionalEncoding3D(embed_dim).cuda()
            self.pos_embd = SinusoidalPositionEmbedding().cuda()
        self.pos_drop = nn.Dropout(p=drop_rate)

        # stochastic depth
        dpr = [x.item() for x in torch.linspace(0, drop_path_rate, sum(depths))]  # stochastic depth decay rule

        # build layers
        self.layers = nn.ModuleList()
        for i_layer in range(self.num_layers):
            layer = BasicLayer(dim=int(embed_dim * 2 ** i_layer),
                                depth=depths[i_layer],
                                num_heads=num_heads[i_layer],
                                window_size=window_size,
                                mlp_ratio=mlp_ratio,
                                qkv_bias=qkv_bias,
                                rpe = rpe,
                                qk_scale=qk_scale,
                                drop=drop_rate,
                                attn_drop=attn_drop_rate,
                                drop_path=dpr[sum(depths[:i_layer]):sum(depths[:i_layer + 1])],
                                norm_layer=norm_layer,
                                downsample=PatchMerging if (i_layer < self.num_layers - 1) else None,
                                use_checkpoint=use_checkpoint,
                               pat_merg_rf=pat_merg_rf,
                               MC_drop=MC_drop)
            self.layers.append(layer)

        num_features = [int(embed_dim * 2 ** i) for i in range(self.num_layers)]
        self.num_features = num_features

        # add a norm layer for each output
        for i_layer in out_indices:
            layer = norm_layer(num_features[i_layer])
            layer_name = f'norm{i_layer}'
            self.add_module(layer_name, layer)

        self._freeze_stages()

    def _freeze_stages(self):
        if self.frozen_stages >= 0:
            self.patch_embed.eval()
            for param in self.patch_embed.parameters():
                param.requires_grad = False

        if self.frozen_stages >= 1 and self.ape:
            self.absolute_pos_embed.requires_grad = False

        if self.frozen_stages >= 2:
            self.pos_drop.eval()
            for i in range(0, self.frozen_stages - 1):
                m = self.layers[i]
                m.eval()
                for param in m.parameters():
                    param.requires_grad = False

    def init_weights(self, pretrained=None):
        """Initialize the weights in backbone.
        Args:
            pretrained (str, optional): Path to pre-trained weights.
                Defaults to None.
        """

        def _init_weights(m):
            if isinstance(m, nn.Linear):
                trunc_normal_(m.weight, std=.02)
                if isinstance(m, nn.Linear) and m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.LayerNorm):
                nn.init.constant_(m.bias, 0)
                nn.init.constant_(m.weight, 1.0)

        if isinstance(pretrained, str):
            self.apply(_init_weights)
        elif pretrained is None:
            self.apply(_init_weights)
        else:
            raise TypeError('pretrained must be a str or None')

    def forward(self, x):
        """Forward function."""
        x = self.patch_embed(x)

        Wh, Ww = x.size(2), x.size(3)
        if self.ape:
            # interpolate the position embedding to the corresponding size
            absolute_pos_embed = nnf.interpolate(self.absolute_pos_embed, size=(Wh, Ww), mode='bicubic')
            x = (x + absolute_pos_embed).flatten(2).transpose(1, 2)  # B Wh*Ww C
        else:
            x = x.flatten(2).transpose(1, 2)
        x = self.pos_drop(x)

        outs = []
        for i in range(self.num_layers):
            layer = self.layers[i]
            x_out, H, W, x, Wh, Ww = layer(x, Wh, Ww)

            if i in self.out_indices:
                norm_layer = getattr(self, f'norm{i}')
                x_out = norm_layer(x_out)

                out = x_out.view(-1, H, W, self.num_features[i]).permute(0, 3, 1, 2).contiguous()
                outs.append(out)
        return outs

    def train(self, mode=True):
        """Convert the model into training mode while keep layers freezed."""
        super(SwinTransformer, self).train(mode)
        self._freeze_stages()

class Conv2dReLU(nn.Sequential):
    def __init__(
            self,
            in_channels,
            out_channels,
            kernel_size,
            padding=0,
            stride=1,
            use_batchnorm=True,
    ):
        conv = nn.Conv2d(
            in_channels,
            out_channels,
            kernel_size,
            stride=stride,
            padding=padding,
            bias=False,
        )
        relu = nn.LeakyReLU(inplace=True)
        if not use_batchnorm:
            nm = nn.InstanceNorm2d(out_channels)
        else:
            nm = nn.BatchNorm2d(out_channels)

        super(Conv2dReLU, self).__init__(conv, nm, relu)


class DecoderBlock(nn.Module):
    def __init__(
            self,
            in_channels,
            out_channels,
            skip_channels=0,
            use_batchnorm=True,
    ):
        super().__init__()
        self.conv1 = Conv2dReLU(
            in_channels + skip_channels,
            out_channels,
            kernel_size=3,
            padding=1,
            use_batchnorm=use_batchnorm,
        )
        self.conv2 = Conv2dReLU(
            out_channels,
            out_channels,
            kernel_size=3,
            padding=1,
            use_batchnorm=use_batchnorm,
        )
        self.up = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=False)

    def forward(self, x, skip=None):
        x = self.up(x)
        if skip is not None:
            x = torch.cat([x, skip], dim=1)
        x = self.conv1(x)
        x = self.conv2(x)
        return x

# Your RegistrationHead class
class RegistrationHead(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size=3, upsampling=1, num_lev=5):
        """
        Args:
            in_channels (int): Number of input channels.
            out_channels (int): Number of output channels for the conv layer.
            kernel_size (int): Kernel size for the convolution.
            upsampling (int): Upsampling factor (if > 1, a nn.Upsample is applied).
            num_lev (int): Number of pyramid levels.
        """
        super(RegistrationHead, self).__init__()
        
        # A simple convolution layer with weight initialization for mean.
        self.conv2d_mean = nn.Conv2d(in_channels, out_channels, kernel_size=kernel_size, 
                                padding=kernel_size // 2)
        # Initialize weights and biases (you can also use nn.init if preferred)
        self.conv2d_mean.weight = nn.Parameter(torch.randn_like(self.conv2d_mean.weight) * 1e-5)
        self.conv2d_mean.bias = nn.Parameter(torch.zeros_like(self.conv2d_mean.bias))
        
        # A simple convolution layer with weight initialization for std.
        self.conv2d_std = nn.Conv2d(in_channels, out_channels, kernel_size=kernel_size, 
                                padding=kernel_size // 2)
        # Initialize weights and biases (you can also use nn.init if preferred)
        self.conv2d_std.weight = nn.Parameter(torch.randn_like(self.conv2d_std.weight) * 1e-5)
        self.conv2d_std.bias = nn.Parameter(torch.zeros_like(self.conv2d_std.bias))
        
        # Optional upsampling layer (if upsampling > 1)
        self.upsampling = (nn.Upsample(scale_factor=upsampling, mode='bilinear', 
                                       align_corners=False)
                           if upsampling > 1 else None)
        
        self.num_lev = num_lev
        
        # Instantiate the classification score module.
        # Note: We assume that the feature maps in the pyramid have 'in_channels' channels.
        self.class_mean_pred = ClassScore(in_channels, hidden_dim=2048, dropout_rate=0.5, num_classes=1)
        self.class_std_pred = ClassScore(in_channels, hidden_dim=2048, dropout_rate=0.5, num_classes=1)
        self.class_score = ClassScore(in_channels, hidden_dim=2048, dropout_rate=0.5, num_classes=1)

    def forward(self, x, y, val):
        # Optionally, you might want to apply self.conv2d or self.upsampling to x first.
        # For example:
        # x = self.conv2d(x)
        # if self.upsampling is not None:
        #     x = self.upsampling(x)
        
        alpha = 0.5
        beta  = 0.5
        
        mean_pred_x = self.conv2d_mean(x)
        if mean_pred_x.requires_grad:
            mean_pred_x.register_hook(lambda grad: grad * alpha)
        mean_pred_x = self.class_mean_pred(mean_pred_x)
            
        mean_pred_y = self.conv2d_mean(y)
        if mean_pred_y.requires_grad:
            mean_pred_y.register_hook(lambda grad: grad * beta)
        mean_pred_y = self.class_mean_pred(mean_pred_y)
            
        # Weighted sum of the two predictions
        mean_pred = 0.75 * mean_pred_x + 0.25 * mean_pred_y
            
        std_pred_x  = self.conv2d_std(x)
        if std_pred_x.requires_grad:
            std_pred_x.register_hook(lambda grad: grad * alpha)
        std_pred_x  = self.class_std_pred(std_pred_x)
            
        std_pred_y  = self.conv2d_std(y)
        if std_pred_y.requires_grad:
            std_pred_y.register_hook(lambda grad: grad * beta)
        std_pred_y  = self.class_std_pred(std_pred_y)
            
        # Weighted sum of the two predictions
        std_pred = 0.75 * std_pred_x + 0.25 * std_pred_y

        if self.training or val:
        
            # Create a pyramid of feature maps
            pyramid_x = self.create_pyramid(x)
            pyramid_y = self.create_pyramid(y)
            
            # Apply the classification head to each pyramid level.
            # This returns a list of classification scores (one per level)
            class_scores_x = [self.class_score(level) for level in pyramid_x]
            class_scores_y = [self.class_score(level) for level in pyramid_y]
            
            # Stack the scores into a tensor of shape: [num_levels, batch_size, 1]
            stacked_scores_x = torch.stack(class_scores_x, dim=0)
            stacked_scores_y = torch.stack(class_scores_y, dim=0)
        
            # Concatenate along the pyramid level dimension (dim=0)
            stacked_scores_combined = torch.cat([stacked_scores_x, stacked_scores_y], dim=0)

            # Compute the mean over the combined pyramid dimension
            mean_scores = torch.mean(stacked_scores_combined, dim=0)  # Shape: [batch_size, 1]

            # Compute the standard deviation over the combined pyramid dimension
            std_scores = torch.std(stacked_scores_combined, dim=0)    # Shape: [batch_size, 1]
        
            # Optionally, if you need probabilities instead of logits, apply sigmoid:
            # prob_scores = torch.sigmoid(mean_scores)
        
            return mean_pred, std_pred, mean_scores, std_scores
        else:
            mean_pred = self.conv2d_mean(x)
            mean_pred = self.class_mean_pred(mean_pred)
            
            std_pred  = self.conv2d_std(x)
            std_pred  = self.class_std_pred(std_pred)
            
            mean_scores = None
            std_scores  = None
            
            return mean_pred, std_pred, mean_scores, std_scores

    def create_pyramid(self, x):
        """
        Creates a pyramid of feature maps.
        
        For each level i, this function downsamples by a factor of 2**i and upsamples by 2**i.
        You can adjust the logic depending on whether you want a pyramid of downsampled maps,
        upsampled maps, or a combination of both.
        """
        pyramid = [x]
        for i in range(self.num_lev):  # Pyramid levels starting at 0
            # Compute scale factors: downsampling factor is the inverse of upsampling.
            scale_factor_d = 1 / (2 ** i)
            scale_factor_u = 2 ** i
            
            # Downsample the feature map
            downsampled_x = nnf.interpolate(x, scale_factor=scale_factor_d, mode='bilinear', 
                                          align_corners=False)
            # Check if the downsampled size is valid (non-zero spatial dimensions)
            if downsampled_x.shape[2] == 0 or downsampled_x.shape[3] == 0:
                break
            
            # Upsample the feature map
            upsampled_x = nnf.interpolate(x, scale_factor=scale_factor_u, mode='bilinear', 
                                        align_corners=False)
            
            # Append one or both of the scaled versions to the pyramid.
            # Here we append the upsampled version as an example.
            pyramid.append(upsampled_x)
            
        return pyramid
        
# Define the module that computes the classification score for a single pyramid level.
class ClassScore(nn.Module):
    def __init__(self, in_channels, hidden_dim=2048, dropout_rate=0.5, num_classes=1):
        """
        Args:
            in_channels (int): Number of input channels (from the feature map).
            hidden_dim (int): Number of units in the hidden fully connected layer.
            dropout_rate (float): Dropout probability.
            num_classes (int): Number of output classes (1 for binary classification).
        """
        super(ClassScore, self).__init__()
        # Global Average Pooling reduces H x W to 1 x 1
        self.gap = nn.AdaptiveAvgPool2d(1)
        # Flatten the pooled features into a vector of size (in_channels)
        self.flatten = nn.Flatten()
        # Fully connected layer: from the number of channels to hidden_dim
        self.fc = nn.Linear(in_channels, hidden_dim)
        # Dropout layer for regularization
        self.dropout = nn.Dropout(dropout_rate)
        # Final classification layer: from hidden_dim to num_classes
        self.classifier = nn.Linear(hidden_dim, num_classes)

    def forward(self, x):
        # x: [batch_size, in_channels, H, W]
        x = self.gap(x)            # -> [batch_size, in_channels, 1, 1]
        x = self.flatten(x)        # -> [batch_size, in_channels]
        x = self.fc(x)             # -> [batch_size, hidden_dim]
        x = self.dropout(x)        # -> [batch_size, hidden_dim]
        x = self.classifier(x)     # 
        return x

class SpatialTransformer(nn.Module):
    """
    N-D Spatial Transformer
    Obtained from https://github.com/voxelmorph/voxelmorph
    """

    def __init__(self, size, mode='bilinear'):
        super().__init__()

        self.mode = mode

        # create sampling grid
        vectors = [torch.arange(0, s) for s in size]
        grids = torch.meshgrid(vectors)
        grid = torch.stack(grids)
        grid = torch.unsqueeze(grid, 0)
        grid = grid.type(torch.FloatTensor)

        # registering the grid as a buffer cleanly moves it to the GPU, but it also
        # adds it to the state dict. this is annoying since everything in the state dict
        # is included when saving weights to disk, so the model files are way bigger
        # than they need to be. so far, there does not appear to be an elegant solution.
        # see: https://discuss.pytorch.org/t/how-to-register-buffer-without-polluting-state-dict
        self.register_buffer('grid', grid)

    def forward(self, src, flow):
        # new locations
        new_locs = self.grid + flow
        shape = flow.shape[2:]

        # need to normalize grid values to [-1, 1] for resampler
        for i in range(len(shape)):
            new_locs[:, i, ...] = 2 * (new_locs[:, i, ...] / (shape[i] - 1) - 0.5)

        # move channels dim to last position
        # also not sure why, but the channels need to be reversed
        if len(shape) == 2:
            new_locs = new_locs.permute(0, 2, 3, 1)
            new_locs = new_locs[..., [1, 0]]
        elif len(shape) == 3:
            new_locs = new_locs.permute(0, 2, 3, 4, 1)
            new_locs = new_locs[..., [2, 1, 0]]

        return nnf.grid_sample(src, new_locs, align_corners=True, mode=self.mode)

class UASTNet(nn.Module):
    def __init__(self, config):
        super(UASTNet, self).__init__()
        self.if_convskip = config.if_convskip
        self.if_transskip = config.if_transskip
        embed_dim = config.embed_dim
        self.transformer = SwinTransformer(patch_size=config.patch_size,
                                           in_chans=config.in_chans,
                                           embed_dim=config.embed_dim,
                                           depths=config.depths,
                                           num_heads=config.num_heads,
                                           window_size=config.window_size,
                                           mlp_ratio=config.mlp_ratio,
                                           qkv_bias=config.qkv_bias,
                                           drop_rate=config.drop_rate,
                                           drop_path_rate=config.drop_path_rate,
                                           ape=config.ape,
                                           spe=config.spe,
                                           rpe=config.rpe,
                                           patch_norm=config.patch_norm,
                                           use_checkpoint=config.use_checkpoint,
                                           out_indices=config.out_indices,
                                           pat_merg_rf=config.pat_merg_rf,
                                           MC_drop=config.MC_drop)
        self.up0 = DecoderBlock(embed_dim*8, embed_dim*4, skip_channels=embed_dim*4 if self.if_transskip else 0, use_batchnorm=False)
        self.up1 = DecoderBlock(embed_dim*4, embed_dim*2, skip_channels=embed_dim*2 if self.if_transskip else 0, use_batchnorm=False)
        self.up2 = DecoderBlock(embed_dim*2, embed_dim, skip_channels=embed_dim if self.if_transskip else 0, use_batchnorm=False)
        self.up3 = DecoderBlock(embed_dim, embed_dim//2, skip_channels=embed_dim//2 if self.if_convskip else 0, use_batchnorm=False)
        self.up4 = DecoderBlock(embed_dim//2, config.reg_head_chan, skip_channels=config.reg_head_chan if self.if_convskip else 0, use_batchnorm=False)
        self.c1 = Conv2dReLU(3, embed_dim//2, 3, 1, use_batchnorm=False)
        self.c2 = Conv2dReLU(3, config.reg_head_chan, 3, 1, use_batchnorm=False)
        self.reg_head = RegistrationHead(in_channels=config.reg_head_chan, out_channels=16, kernel_size=3)
        self.avg_pool = nn.AvgPool2d(3, stride=2, padding=1)

        self._dropout_T = 1000 #25
        self._dropout_p = 0.5

    def forward(self, x, disp=None,  val=True, mc_dropout=False, test=False):
        alpha = 0.5
        beta  = 0.5
        source = x[:, 0:3, :, :]
        colorSeg = x[:, 3:6, :, :]
        if disp is not None:
            dispSrc = disp[:, 0:1, :, :]
        if self.if_convskip:
            x_s0 = source.clone()
            x_s1 = self.avg_pool(source)
            f4 = self.c1(x_s1)
            if f4.requires_grad:
                f4.register_hook(lambda grad: grad * alpha)
            f4 = nnf.dropout3d(f4, self._dropout_p, training=self.training)
            f5 = self.c2(x_s0)
            if f5.requires_grad:
                f5.register_hook(lambda grad: grad * alpha)
            f5 = nnf.dropout3d(f5, self._dropout_p, training=self.training)
        else:
            f4 = None
            f5 = None

        out_feats = self.transformer(source)

        if self.if_transskip:
            f1 = out_feats[-2]
            f2 = out_feats[-3]
            f3 = out_feats[-4]
        else:
            f1 = None
            f2 = None
            f3 = None
        source = self.up0(out_feats[-1], f1)
        if source.requires_grad:
            source.register_hook(lambda grad: grad * alpha)
        source = self.up1(source, f2)
        if source.requires_grad:
            source.register_hook(lambda grad: grad * alpha)
        source = self.up2(source, f3)
        if source.requires_grad:
            source.register_hook(lambda grad: grad * alpha)
        source = self.up3(source, f4)
        if source.requires_grad:
            source.register_hook(lambda grad: grad * alpha)
        source1 = self.up4(source, f5)
        if source1.requires_grad:
            source1.register_hook(lambda grad: grad * alpha)
        
        # print(f"The shape of source in registration head is: {source1.shape}")
        
        
        if self.if_convskip:
            x_s0_2 = colorSeg.clone()
            x_s1_2 = self.avg_pool(colorSeg)
            f4_2 = self.c1(x_s1_2)
            if f4_2.requires_grad:
                f4_2.register_hook(lambda grad: grad * beta)
            f4_2 = nnf.dropout3d(f4_2, self._dropout_p, training=self.training)
            f5_2 = self.c2(x_s0_2)
            if f5_2.requires_grad:
                f5_2.register_hook(lambda grad: grad * beta)
            f5_2 = nnf.dropout3d(f5_2, self._dropout_p, training=self.training)
        else:
            f4_2 = None
            f5_2 = None

        out_feats_2 = self.transformer(colorSeg)

        if self.if_transskip:
            f1_2 = out_feats_2[-2]
            f2_2 = out_feats_2[-3]
            f3_2 = out_feats_2[-4]
        else:
            f1_2 = None
            f2_2 = None
            f3_2 = None
        colorSeg_2 = self.up0(out_feats_2[-1], f1_2)
        if colorSeg_2.requires_grad:
            colorSeg_2.register_hook(lambda grad: grad * beta)
        colorSeg_2 = self.up1(colorSeg_2, f2_2)
        if colorSeg_2.requires_grad:
            colorSeg_2.register_hook(lambda grad: grad * beta)
        colorSeg_2 = self.up2(colorSeg_2, f3_2)
        if colorSeg_2.requires_grad:
            colorSeg_2.register_hook(lambda grad: grad * beta)
        colorSeg_2 = self.up3(colorSeg_2, f4_2)
        if colorSeg_2.requires_grad:
            colorSeg_2.register_hook(lambda grad: grad * beta)
        colorSeg_21 = self.up4(colorSeg_2, f5_2)
        if colorSeg_21.requires_grad:
            colorSeg_21.register_hook(lambda grad: grad * beta)
        
        # print(f"The shape of segmented color map in registration head is: {colorSeg_21.shape}")
        
        if test == False:
            mean_pred, std_pred, mean_scores, var_scores = self.reg_head(source1, colorSeg_21, val)
            return mean_pred, std_pred, mean_scores, var_scores
        else:
            mean_pred, std_pred, _, _ = self.reg_head(source1, colorSeg_21, val=False)
            return mean_pred, std_pred, None, None
CONFIGS = {
    'UASTNet': configs.get_2DUASTNet_config(),
}
