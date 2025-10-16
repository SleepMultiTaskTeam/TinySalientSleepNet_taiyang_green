# --------------------------------------------------------
# Swin Transformer
# Copyright (c) 2021 Microsoft
# Licensed under The MIT License [see LICENSE for details]
# Written by Ze Liu
# --------------------------------------------------------

import torch
import torch.nn as nn
import torch.utils.checkpoint as checkpoint
# from einops import rearrange
from model.my_trunc_normal import trunc_normal_
# from timm.models.layers import DropPath, to_2tuple, trunc_normal_
# from torchsummary import summary
from model.mygelu import GELU

class Mlp(nn.Module):
    def __init__(self, in_features, hidden_features=None, out_features=None, act_layer=GELU, drop=0.):
        super().__init__()
        out_features = out_features or in_features
        hidden_features = hidden_features or in_features
        self.fc1 = nn.Linear(in_features, hidden_features)
        self.act = act_layer()
        self.fc2 = nn.Linear(hidden_features, out_features)
        self.drop = nn.Dropout(drop)

    def forward(self, x):
        x = self.fc1(x)
        x = self.act(x)
        x = self.drop(x)
        x = self.fc2(x)
        x = self.drop(x)
        return x

# 把图片划分windows，本来第一个维度是batch，但是把windows也合并到了第一个维度，但是应该是线性排列可以复原
def window_partition_1d(x, window_size):
    """
    Args:
        x: (B, W, C)
        window_size (int): window size

    Returns:
        windows: (num_windows*B, window_size, C)
    """
    B, W, C = x.shape
    # 单独看通道W，把W这个一维的东西分成二维的W // window_size, window_size的矩阵，1d的更好理解
    x = x.view(B, W // window_size, window_size, C)
    # 原版对二维操作复杂，但是1维就直接和前面拼起来就行
    # 使用permute和transpose调换后内存不连续，不能使用view，需要使用contiguous
    # 1维不需要比调换轴，直接view，把前两个维度合并
    windows = x.view(-1, window_size, C)  # B*nW, window_size, C
    return windows

# 1d的恢复窗口函数
def window_reverse_1d(windows, window_size, W):
    """
    Args:
        windows: (num_windows*B, window_size, window_size, C)
        window_size (int): Window size
        H (int): Height of image
        W (int): Width of image

    Returns:
        x: (B, H, W, C)
    """
    # 计算原来的batch大小,用第一维度除以窗口个数
    B = int(windows.shape[0] / (W / window_size))
    # 像之前划分窗口一样逐渐回复维度
    # 分出2个维度
    x = windows.view(B, W // window_size, window_size, -1)
    # 输出1d
    x = x.view(B, W, -1)
    # x = x.permute(0, 1, 3, 2, 4, 5).contiguous().view(B, H, W, -1)
    return x

# transformer
# 注意dim是channel，是最后一个维度
# 这个模块只管attention和mask，不考虑偏移，因为偏移不偏移都是按直接滑窗的方式进行attention
class WindowAttention_1d(nn.Module):
    r""" Window based multi-head self attention (W-MSA) module with relative position bias.
    It supports both of shifted and non-shifted window.

    Args:
        dim (int): Number of input channels.
        window_size (tuple[int]): The height and width of the window.
        num_heads (int): Number of attention heads.
        qkv_bias (bool, optional):  If True, add a learnable bias to query, key, value. Default: True
        qk_scale (float | None, optional): Override default qk scale of head_dim ** -0.5 if set
        attn_drop (float, optional): Dropout ratio of attention weight. Default: 0.0
        proj_drop (float, optional): Dropout ratio of output. Default: 0.0
    """

    def __init__(self, dim, window_size_1d, num_heads, qkv_bias=True, qk_scale=None, attn_drop=0., proj_drop=0.):
        # 尝试直接h维度给1计算相对位置编码
        window_size = (1, window_size_1d)
        super().__init__()
        self.dim = dim
        self.window_size = window_size  # Wh, Ww
        self.num_heads = num_heads
        head_dim = dim // num_heads
        self.scale = qk_scale or head_dim ** -0.5

        # 相对位置编码表，可学习变量
        # define a parameter table of relative position bias
        self.relative_position_bias_table = nn.Parameter(
            torch.zeros((2 * window_size[0] - 1) * (2 * window_size[1] - 1), num_heads))  # 2*Wh-1 * 2*Ww-1, nH

        # 获取相对位置编码
        # get pair-wise relative position index for each token inside the window
        coords_h = torch.arange(self.window_size[0]) # 获取一个0，1，2，3....到window_size-1的序列
        coords_w = torch.arange(self.window_size[1])
        coords = torch.stack(torch.meshgrid([coords_h, coords_w]))  # 2, Wh, Ww
        coords_flatten = torch.flatten(coords, 1)  # 2, Wh*Ww
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
        """
        Args:
            x: input features with shape of (num_windows*B, N, C)
            mask: (0/-inf) mask with shape of (num_windows, Wh*Ww, Wh*Ww) or None
        """
        B_, N, C = x.shape
        #linear 分出kqv，然后再切分多头
        qkv = self.qkv(x).reshape(B_, N, 3, self.num_heads, C // self.num_heads).permute(2, 0, 3, 1, 4)  # 注kqv, batch, head, N, C//head
        q, k, v = qkv[0], qkv[1], qkv[2]  # make torchscript happy (cannot use tensor as tuple)

        # 进行multihead attention
        q = q * self.scale
        attn = (q @ k.transpose(-2, -1))  # 可以看作前面维度都不变，就最后两个维度作为矩阵相乘

        # 获取相对位置编码并加到attn上
        # todo:回头可以自己写个1d的相对位置编码
        relative_position_bias = self.relative_position_bias_table[self.relative_position_index.view(-1)].view(
            self.window_size[0] * self.window_size[1], self.window_size[0] * self.window_size[1], -1)  # Wh*Ww,Wh*Ww,nH
        relative_position_bias = relative_position_bias.permute(2, 0, 1).contiguous()  # nH, Wh*Ww, Wh*Ww
        attn = attn + relative_position_bias.unsqueeze(0)

        if mask is not None:
            nW = mask.shape[0]
            # 这里又把第nW * B维度打回了原来的两个维度，加mask
            # 输入的X是B_, N, C，这里不用考虑1D还是2D，2D输入进来的N是H * W，1D就是W，比如N是49，attention矩阵是N * N
            # 输入的mask是(nW, 49*49)
            attn = attn.view(B_ // nW, nW, self.num_heads, N, N) + mask.unsqueeze(1).unsqueeze(0)
            # 加完再变回 nW * B
            attn = attn.view(-1, self.num_heads, N, N)
            attn = self.softmax(attn)
        else:
            attn = self.softmax(attn)

        attn = self.attn_drop(attn)
        # 加权
        x = (attn @ v).transpose(1, 2).reshape(B_, N, C)
        x = self.proj(x)
        x = self.proj_drop(x)
        return x

    def extra_repr(self) -> str:
        return f'dim={self.dim}, window_size={self.window_size}, num_heads={self.num_heads}'

    def flops(self, N):
        # calculate flops for 1 window with token length of N
        flops = 0
        # qkv = self.qkv(x)
        flops += N * self.dim * 3 * self.dim
        # attn = (q @ k.transpose(-2, -1))
        flops += self.num_heads * N * (self.dim // self.num_heads) * N
        #  x = (attn @ v)
        flops += self.num_heads * N * N * (self.dim // self.num_heads)
        # x = self.proj(x)
        flops += N * self.dim * self.dim
        return flops


class SwinTransformerBlock(nn.Module):
    r""" Swin Transformer Block.

    Args:
        dim (int): Number of input channels.
        input_resolution (int): Input resulotion.
        num_heads (int): Number of attention heads.
        window_size (int): Window size.
        shift_size (int): Shift size for SW-MSA.
        mlp_ratio (float): Ratio of mlp hidden dim to embedding dim.
        qkv_bias (bool, optional): If True, add a learnable bias to query, key, value. Default: True
        qk_scale (float | None, optional): Override default qk scale of head_dim ** -0.5 if set.
        drop (float, optional): Dropout rate. Default: 0.0
        attn_drop (float, optional): Attention dropout rate. Default: 0.0
        drop_path (float, optional): Stochastic depth rate. Default: 0.0
        act_layer (nn.Module, optional): Activation layer. Default: nn.GELU
        norm_layer (nn.Module, optional): Normalization layer.  Default: nn.LayerNorm
    """

    def __init__(self, dim, input_resolution, num_heads, window_size=7, shift_size=0,
                 mlp_ratio=4., qkv_bias=True, qk_scale=None, drop=0., attn_drop=0., drop_path=0.,
                 act_layer=GELU, norm_layer=nn.LayerNorm):
        super().__init__()
        self.dim = dim
        # 这里设定input_resolution为1d
        self.input_resolution = input_resolution
        self.num_heads = num_heads
        self.window_size = window_size
        self.shift_size = shift_size
        self.mlp_ratio = mlp_ratio
        if self.input_resolution <= self.window_size:
            # if window size is larger than input resolution, we don't partition windows
            self.shift_size = 0
            self.window_size = self.input_resolution
        assert 0 <= self.shift_size < self.window_size, "shift_size must in 0-window_size"

        self.norm1 = norm_layer(dim)
        # 因为默认窗口大小是方形，我们这里是1d的，所以把输入改成(1, sequence_size)
        self.attn = WindowAttention_1d(
            dim, window_size_1d=self.window_size, num_heads=num_heads,
            qkv_bias=qkv_bias, qk_scale=qk_scale, attn_drop=attn_drop, proj_drop=drop)

        #self.drop_path = DropPath(drop_path) if drop_path > 0. else nn.Identity()
        self.drop_path = nn.Identity()
        self.norm2 = norm_layer(dim)
        mlp_hidden_dim = int(dim * mlp_ratio)
        self.mlp = Mlp(in_features=dim, hidden_features=mlp_hidden_dim, act_layer=act_layer, drop=drop)

        #计算mask，遮住两个不相邻边做attn的位置，每个窗都有一个对应mask
        if self.shift_size > 0:
            # calculate attention mask for SW-MSA
            # H改成1，W用尺寸
            H = 1
            W = self.input_resolution
            img_mask = torch.zeros((1, W, 1))  # 1 H W 1
            # slice(0, -self.window_size)是空出最后的window大小裁切（完美空出）
            # slice(-self.window_size, -self.shift_size)是从最后window开始裁切到滑动块之前
            # slice(-self.shift_size, None)是把最后的shift_size切出来
            # 整个就是把长度分成了这三段
            # 在图像平移后也正好是这三段，第一段就没变化的，后面两段是正好在那个错位的窗口，是两个互相不能做att的段
            w_slices = (slice(0, -self.window_size),
                        slice(-self.window_size, -self.shift_size),
                        slice(-self.shift_size, None))
            cnt = 0
            # 在对应区段打标记，标记逐块递增，3*3总共9个标记
            # 改成1d的就是3个标记
            for w in w_slices:
                img_mask[:, w, :] = cnt
                cnt += 1

            # 滑窗分割，窗虽然不只有9*9个，但是涉及边缘问题的窗确实只需要上面的那几类标记，不用每个窗都单独分出它的标记
            # 按窗排好，输出为，第一维度：窗口个数，窗口长，窗口宽，1
            mask_windows = window_partition_1d(img_mask, self.window_size)  # nW, window_size, 1
            # 维度重构，长宽合并成一条线，比如7*7的窗是49的一条线，线的每个值是上面打过的标记，0-8
            mask_windows = mask_windows.view(-1, self.window_size)
            # 49和49做self attn形成49*49的加权矩阵
            # unsqueeze：增加维度，比如之前是(nW, 49),在第一维度加就是(nW, 1, 49)，在第二维度加就是(nW, 49, 1)
            # 如果两个地方的标记是相同的，比如都是5，5-5 = 0，其它情况都不是0，就是说其他情况都是无效的
            attn_mask = mask_windows.unsqueeze(1) - mask_windows.unsqueeze(2)
            # 所以mask给非0的加-100，是0的就不mask
            attn_mask = attn_mask.masked_fill(attn_mask != 0, float(-100.0)).masked_fill(attn_mask == 0, float(0.0))
        else:
            attn_mask = None

        self.register_buffer("attn_mask", attn_mask)

    def forward(self, x):
        # 改成1d
        W = self.input_resolution
        B, L, C = x.shape
        assert L == W, "input feature has wrong size"

        shortcut = x
        x = self.norm1(x)
        # 1d，不需要换维度
        # x = x.view(B, H, W, C)

        # cyclic shift
        if self.shift_size > 0:
            # 循环移位，1d版
            shifted_x = torch.roll(x, shifts=-self.shift_size, dims=1)
        else:
            shifted_x = x
        # todo:检查循环移位正确性
        # partition windows
        x_windows = window_partition_1d(shifted_x, self.window_size)  # nW*B, window_size, C
        # 1d不需要 x_windows = x_windows.view(-1, self.window_size * self.window_size, C)  # nW*B, window_size*window_size, C

        # W-MSA/SW-MSA
        # X这里就是nW*B, window_size, C
        # attn_mask的维度是Wn, win_size, C，第一维没有乘batch
        attn_windows = self.attn(x_windows, mask=self.attn_mask)  # nW*B, window_size, C

        # window_reverse是window_partition的逆过程，把划分到第一维度的nW * B中的窗口还原
        # merge windows
        # 2d的需要先把尺寸拓宽，1d不需要
        # 也就是进出这些window_reverse和partition的时候作者都进行了1维分成2维，2维再转回一维的操作
        # attn_windows = attn_windows.view(-1, self.window_size, self.window_size, C)
        shifted_x = window_reverse_1d(attn_windows, self.window_size, W)  # B W' C

        # 因为之前循环位移了，这里要位移回去
        # reverse cyclic shift
        if self.shift_size > 0:
            # 同样是1d的
            x = torch.roll(shifted_x, shifts=self.shift_size, dims=1)
        else:
            x = shifted_x
        # 在attention后把X从图像状态变回来
        # x = x.view(B, H * W, C)

        # FFN
        # shortcut是残差，这里两行代码一行一个残差
        x = shortcut + self.drop_path(x)
        x = x + self.drop_path(self.mlp(self.norm2(x)))

        return x

    # def extra_repr(self) -> str:
    #     return f"dim={self.dim}, input_resolution={self.input_resolution}, num_heads={self.num_heads}, " \
    #            f"window_size={self.window_size}, shift_size={self.shift_size}, mlp_ratio={self.mlp_ratio}"
    #
    # def flops(self):
    #     flops = 0
    #     H, W = self.input_resolution
    #     # norm1
    #     flops += self.dim * H * W
    #     # W-MSA/SW-MSA
    #     nW = H * W / self.window_size / self.window_size
    #     flops += nW * self.attn.flops(self.window_size * self.window_size)
    #     # mlp
    #     flops += 2 * H * W * self.dim * self.dim * self.mlp_ratio
    #     # norm2
    #     flops += self.dim * H * W
    #     return flops

# 在token上工作（我们之前嵌入的每个序列都是一个token）
# 4个token拼起来，其实就是把滑窗4个一组拼起来（不过嵌入过了）
# 输入：B, H, C， channel在第三维度（embed的输出是B, token个数, token序列），这里token个数其实就是平铺的图像，token序列是图像的信息（channel）
# B,H,C是对于图像来说的，H是H*W，C是特征channel，B,C,H是对于token来说的，C是token个数，H是一个token的长度
# 输出：B, H/2, C
class PatchMerging_1d(nn.Module):
    r""" Patch Merging Layer.

    Args:
        input_resolution (tuple[int]): Resolution of input feature.
        dim (int): Number of input channels.
        norm_layer (nn.Module, optional): Normalization layer.  Default: nn.LayerNorm
    """
# 用sequence_size替代input_resolution
    def __init__(self, sequence_size, dim, norm_layer=nn.LayerNorm):
        super().__init__()
        self.sequence_size = sequence_size
        self.dim = dim
        self.reduction = nn.Linear(2 * dim, 2 * dim, bias=False)
        self.norm = norm_layer(2 * dim)
        # self.reduction = nn.Linear(4 * dim, 2 * dim, bias=False)
        # self.norm = norm_layer(4 * dim)

    def forward(self, x):
        """
        x: B, H*W, C
        """
        # H = self.sequence_size
        B, L, C = x.shape
        # assert L == H * W, "input feature has wrong size"
        # assert H % 2 == 0 and W % 2 == 0, f"x size ({H}*{W}) are not even."

        # x = x.view(B, H, W, C) #不需要扩展维度，我们本来就是1维的

        # 它是用4个不同的起始点加步幅采集一个窗内不同的角
        # x0 = x[:, 0::2, 0::2, :]  # B H/2 W/2 C
        # x1 = x[:, 1::2, 0::2, :]  # B H/2 W/2 C
        # x2 = x[:, 0::2, 1::2, :]  # B H/2 W/2 C
        # x3 = x[:, 1::2, 1::2, :]  # B H/2 W/2 C

        x0 = x[:, 0::2, :]
        x1 = x[:, 1::2, :]
        # 然后把4个角（这里是2个角）拼起来
        # 本来4个拼起来通道数会多4倍，我这里只能多2倍
        # 其实我可以设置成任意倍率（就是控制拼接点的个数呗），但是60000除以4最多除以2次，然后就会出分数，所以降采样2倍比较好调
        x = torch.cat([x0, x1], -1)  # B H/2 2*C
        x = x.view(B, -1, 2 * C)  # B H/2 2*C
        # linear只会操作最后一个维度
        x = self.norm(x)
        x = self.reduction(x)

        return x

    # def extra_repr(self) -> str:
    #     return f"input_resolution={self.input_resolution}, dim={self.dim}"
    #
    # def flops(self):
    #     H, W = self.input_resolution
    #     flops = H * W * self.dim
    #     flops += (H // 2) * (W // 2) * 4 * self.dim * 2 * self.dim
    #     return flops

class PatchExpand_1d(nn.Module):
    def __init__(self, dim, dim_scale=2, norm_layer=nn.LayerNorm):
        super().__init__()
        # self.input_resolution = input_resolution
        self.dim = dim
        # 因为1d图缩小一倍不需要expand，所以可以考虑去掉
        # self.expand = nn.Linear(dim, dim, bias=False) if dim_scale==2 else nn.Identity()
        self.norm = norm_layer(dim // dim_scale)

    def forward(self, x):
        """
        x: B, H*W, C
        """
        # W = self.input_resolution
        # x = self.expand(x)
        B, L, C = x.shape
        # assert L == H * W, "input feature has wrong size"

        # x = x.view(B, H, W, C)
        # 对于2d的数据rearrange是把通道C，分解成2*2*c=C，然后补充到前面的维度中
        # x = rearrange(x, 'b h w (p1 p2 c)-> b (h p1) (w p2) c', p1=2, p2=2, c=C//4)
        # x = rearrange(x, 'b l (p1 c)-> b (l p1) c', p1=2, c=C // 2)
        x = torch.reshape(x, (B, L*2, C//2))

        # x = x.view(B,-1,C//4)
        x = self.norm(x)

        return x


class BasicLayer(nn.Module):
    """ A basic Swin Transformer layer for one stage.

    Args:
        dim (int): Number of input channels.
        input_resolution (tuple[int]): Input resolution.
        depth (int): Number of blocks.
        num_heads (int): Number of attention heads.
        window_size (int): Local window size.
        mlp_ratio (float): Ratio of mlp hidden dim to embedding dim.
        qkv_bias (bool, optional): If True, add a learnable bias to query, key, value. Default: True
        qk_scale (float | None, optional): Override default qk scale of head_dim ** -0.5 if set.
        drop (float, optional): Dropout rate. Default: 0.0
        attn_drop (float, optional): Attention dropout rate. Default: 0.0
        drop_path (float | tuple[float], optional): Stochastic depth rate. Default: 0.0
        norm_layer (nn.Module, optional): Normalization layer. Default: nn.LayerNorm
        downsample (nn.Module | None, optional): Downsample layer at the end of the layer. Default: None
        use_checkpoint (bool): Whether to use checkpointing to save memory. Default: False.
    """

    def __init__(self, dim, input_resolution, depth, num_heads, window_size,
                 mlp_ratio=4., qkv_bias=True, qk_scale=None, drop=0., attn_drop=0.,
                 drop_path=0., norm_layer=nn.LayerNorm, downsample=None, use_checkpoint=False):

        super().__init__()
        self.dim = dim
        self.input_resolution = input_resolution
        self.depth = depth
        self.use_checkpoint = use_checkpoint

        # build blocks
        self.blocks = nn.ModuleList([
            SwinTransformerBlock(dim=dim, input_resolution=input_resolution,
                                 num_heads=num_heads, window_size=window_size,
                                 shift_size=0 if (i % 2 == 0) else window_size // 2,
                                 mlp_ratio=mlp_ratio,
                                 qkv_bias=qkv_bias, qk_scale=qk_scale,
                                 drop=drop, attn_drop=attn_drop,
                                 drop_path=drop_path[i] if isinstance(drop_path, list) else drop_path,
                                 norm_layer=norm_layer)
            for i in range(depth)])

        # patch merging layer
        if downsample is not None:
            self.downsample = downsample(input_resolution, dim=dim, norm_layer=norm_layer)
        else:
            self.downsample = None

    def forward(self, x):
        for blk in self.blocks:
            if self.use_checkpoint:
                x = checkpoint.checkpoint(blk, x)
            else:
                x = blk(x)
        if self.downsample is not None:
            x = self.downsample(x)
        return x

    def extra_repr(self) -> str:
        return f"dim={self.dim}, input_resolution={self.input_resolution}, depth={self.depth}"

    def flops(self):
        flops = 0
        for blk in self.blocks:
            flops += blk.flops()
        if self.downsample is not None:
            flops += self.downsample.flops()
        return flops

# 滑窗嵌入，把窗口内的信息嵌入到另一个一维序列（在图像中不可少，序列里我们这边直接迁移过来）
# 因为有窗口大小，所以输出也起到了降采样的功能
# 输出：channel是token的组数（按二维排列成图片也可以），H是嵌入后的序列
class PatchEmbed_1d(nn.Module):
    r""" Image to Patch Embedding

    Args:
        img_size (int): Image size.  Default: 224.
        patch_size (int): Patch token size. Default: 4.
        in_chans (int): Number of input image channels. Default: 3.
        embed_dim (int): Number of linear projection output channels. Default: 96.
        norm_layer (nn.Module, optional): Normalization layer. Default: None
    """
    # patch_size这里用于降采样，是降采样率
    def __init__(self, sequence_size=60000, patch_size=4, in_chans=2, embed_dim=96, norm_layer=None):
        super().__init__()
        # 降采样后的序列长度 num_patches
        # 用sequence_size 替代原代码中的image_size
        num_patches = sequence_size // patch_size # [img_size[0] // patch_size[0], img_size[1] // patch_size[1]]
        self.sequence_size = sequence_size
        self.patch_size = patch_size
        self.num_patches = num_patches

        self.in_chans = in_chans
        self.embed_dim = embed_dim
        # 一个池化来进行滑窗划分，同时进行嵌入，也就是滑出来的窗直接被嵌入到了embed_dim中，在长度上就是1了，但是channel上变成embed_dim
        self.proj = nn.Conv1d(in_chans, embed_dim, kernel_size=patch_size, stride=patch_size)
        if norm_layer is not None:
            self.norm = norm_layer(embed_dim)
        else:
            self.norm = None

    def forward(self, x):
        B, C, H = x.shape
        # FIXME look at relaxing size constraints
        assert H == self.sequence_size, \
            f"Input image size ({H}) doesn't match model ({self.sequence_size})."
        # 原来要把第二维度展平，这里我们是1d图片就不用了
        # channel放到长度的位置，因为embed的结果上面被放到了channel的位置
        x = self.proj(x).transpose(1, 2)  # B Ph*Pw C
        if self.norm is not None:
            x = self.norm(x)
        return x

    # def flops(self):
    #     Ho, Wo = self.patches_resolution
    #     flops = Ho * Wo * self.embed_dim * self.in_chans * (self.patch_size[0] * self.patch_size[1])
    #     if self.norm is not None:
    #         flops += Ho * Wo * self.embed_dim
    #     return flops


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
        window_size (int): Window size. Default: 7
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

    def __init__(self, img_size=224, patch_size=4, in_chans=3, num_classes=1000,
                 embed_dim=96, depths=[2, 2, 6, 2], num_heads=[3, 6, 12, 24],
                 window_size=7, mlp_ratio=4., qkv_bias=True, qk_scale=None,
                 drop_rate=0., attn_drop_rate=0., drop_path_rate=0.1,
                 norm_layer=nn.LayerNorm, ape=False, patch_norm=True,
                 use_checkpoint=False, **kwargs):
        super().__init__()

        self.num_classes = num_classes
        self.num_layers = len(depths)
        self.embed_dim = embed_dim
        self.ape = ape
        self.patch_norm = patch_norm
        self.num_features = int(embed_dim * 2 ** (self.num_layers - 1))
        self.mlp_ratio = mlp_ratio

        # split image into non-overlapping patches
        self.patch_embed = PatchEmbed(
            img_size=img_size, patch_size=patch_size, in_chans=in_chans, embed_dim=embed_dim,
            norm_layer=norm_layer if self.patch_norm else None)
        num_patches = self.patch_embed.num_patches
        patches_resolution = self.patch_embed.patches_resolution
        self.patches_resolution = patches_resolution

        # absolute position embedding
        if self.ape:
            self.absolute_pos_embed = nn.Parameter(torch.zeros(1, num_patches, embed_dim))
            trunc_normal_(self.absolute_pos_embed, std=.02)

        self.pos_drop = nn.Dropout(p=drop_rate)

        # stochastic depth
        dpr = [x.item() for x in torch.linspace(0, drop_path_rate, sum(depths))]  # stochastic depth decay rule

        # build layers
        self.layers = nn.ModuleList()
        for i_layer in range(self.num_layers):
            layer = BasicLayer(dim=int(embed_dim * 2 ** i_layer),
                               input_resolution=(patches_resolution[0] // (2 ** i_layer),
                                                 patches_resolution[1] // (2 ** i_layer)),
                               depth=depths[i_layer],
                               num_heads=num_heads[i_layer],
                               window_size=window_size,
                               mlp_ratio=self.mlp_ratio,
                               qkv_bias=qkv_bias, qk_scale=qk_scale,
                               drop=drop_rate, attn_drop=attn_drop_rate,
                               drop_path=dpr[sum(depths[:i_layer]):sum(depths[:i_layer + 1])],
                               norm_layer=norm_layer,
                               downsample=PatchMerging if (i_layer < self.num_layers - 1) else None,
                               use_checkpoint=use_checkpoint)
            self.layers.append(layer)

        self.norm = norm_layer(self.num_features)
        self.avgpool = nn.AdaptiveAvgPool1d(1)
        self.head = nn.Linear(self.num_features, num_classes) if num_classes > 0 else nn.Identity()

        self.apply(self._init_weights)

    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            trunc_normal_(m.weight, std=.02)
            if isinstance(m, nn.Linear) and m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.LayerNorm):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)

    @torch.jit.ignore
    def no_weight_decay(self):
        return {'absolute_pos_embed'}

    @torch.jit.ignore
    def no_weight_decay_keywords(self):
        return {'relative_position_bias_table'}

    def forward_features(self, x):
        x = self.patch_embed(x)
        if self.ape:
            x = x + self.absolute_pos_embed
        x = self.pos_drop(x)

        for layer in self.layers:
            x = layer(x)

        x = self.norm(x)  # B L C
        x = self.avgpool(x.transpose(1, 2))  # B C 1
        x = torch.flatten(x, 1)
        return x

    def forward(self, x):
        x = self.forward_features(x)
        x = self.head(x)
        return x

    def flops(self):
        flops = 0
        flops += self.patch_embed.flops()
        for i, layer in enumerate(self.layers):
            flops += layer.flops()
        flops += self.num_features * self.patches_resolution[0] * self.patches_resolution[1] // (2 ** self.num_layers)
        flops += self.num_features * self.num_classes
        return flops

if __name__ == '__main__':
    """ 验证windowAttention无mask的状态是否可正常运行 """
    # windowatt = WindowAttention_1d(dim=100, window_size_1d=10, num_heads=2)
    # #window size就是当前输入的序列长度了
    # input = torch.zeros((16, 10, 100))
    # output = windowatt(input, None)
    # print(output.shape)
    # windowatt.cuda()
    # summary(windowatt, (10, 100))

    """验证升维模块能不能用"""
    # upLayer = PatchExpand_1d(dim=100, dim_scale=2)
    # input = torch.zeros((16, 10, 100))
    # output = upLayer(input)
    # print(output.shape)
    # upLayer.cuda()
    # summary(upLayer, (10, 100))

    # todo:因为我降采样C没变，导致网络加深也没办法让C增加，那里需要重写一下
    """ 验证swinBlock能不能正常运行，无shift_size的 """
    swinBlock = SwinTransformerBlock(dim=300, input_resolution=30000, num_heads=10, window_size=20, shift_size=2)
    input = torch.zeros((16, 30000, 300))
    output = swinBlock(input)
    print(output.size())
    swinBlock.cuda()
    # summary(swinBlock, (30000, 300))