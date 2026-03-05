import torch
import torch.nn as nn


class Conv(torch.nn.Module):

    def __init__(self, in_ch, out_ch, activation: nn.modules.activation, k=1, s=1, p=0, g=1):
        super().__init__()
        self.conv = torch.nn.Conv2d(in_ch, out_ch, k, s, p, groups=g, bias=False)
        self.norm = torch.nn.BatchNorm2d(out_ch, eps=0.001, momentum=0.03)
        self.relu = activation

    def forward(self, x):
        return self.relu(self.norm(self.conv(x)))

    def fuse_forward(self, x):
        return self.relu(self.conv(x))


class Residual(torch.nn.Module):
    def __init__(self, ch, e=0.5):
        super().__init__()
        self.conv1 = Conv(ch, int(ch * e), activation=torch.nn.SiLU(), k=3, p=1)
        self.conv2 = Conv(int(ch * e), ch, activation=torch.nn.SiLU(), k=3, p=1)

    def forward(self, x):
        return x + self.conv2(self.conv1(x))


class CSPModule(torch.nn.Module):
    def __init__(self, in_ch, out_ch):
        super().__init__()
        self.conv1 = Conv(in_ch, out_ch // 2, activation=torch.nn.SiLU())
        self.conv2 = Conv(in_ch, out_ch // 2, activation=torch.nn.SiLU())
        self.conv3 = Conv(2 * (out_ch // 2), out_ch, activation=torch.nn.SiLU())
        self.res_m = torch.nn.Sequential(
            Residual(out_ch // 2, e=1.0),
            Residual(out_ch // 2, e=1.0)
        )

    def forward(self, x):
        y = self.res_m(self.conv1(x))
        return self.conv3(torch.cat((y, self.conv2(x)), dim=1))


class CSP(torch.nn.Module):
    def __init__(self, in_ch, out_ch, n, csp, r):
        super().__init__()
        self.conv1 = Conv(in_ch, 2 * (out_ch // r), activation=torch.nn.SiLU())
        self.conv2 = Conv((2 + n) * (out_ch // r), out_ch, activation=torch.nn.SiLU())

        if not csp:
            self.res_m = torch.nn.ModuleList(Residual(out_ch // r) for _ in range(n))
        else:
            self.res_m = torch.nn.ModuleList(CSPModule(out_ch // r, out_ch // r) for _ in range(n))

    def forward(self, x):
        y = list(self.conv1(x).chunk(2, 1))
        y.extend(m(y[-1]) for m in self.res_m)
        return self.conv2(torch.cat(y, dim=1))


class SPP(torch.nn.Module):
    def __init__(self, in_ch, out_ch, k=5):
        super().__init__()
        self.conv1 = Conv(in_ch, in_ch // 2, activation=torch.nn.SiLU())
        self.conv2 = Conv(in_ch * 2, out_ch, activation=torch.nn.SiLU())
        self.res_m = torch.nn.MaxPool2d(k, stride=1, padding=k // 2)

    def forward(self, x):
        x = self.conv1(x)
        y1 = self.res_m(x)
        y2 = self.res_m(y1)
        return self.conv2(torch.cat(tensors=[x, y1, y2, self.res_m(y2)], dim=1))


class Attention(torch.nn.Module):

    def __init__(self, ch, num_head):
        super().__init__()
        self.num_head = num_head
        self.dim_head = ch // num_head
        self.dim_key = self.dim_head // 2
        self.scale = self.dim_key ** -0.5

        self.qkv = Conv(ch, ch + self.dim_key * num_head * 2, activation=torch.nn.Identity())

        self.conv1 = Conv(ch, ch, activation=torch.nn.Identity(), k=3, p=1, g=ch)
        self.conv2 = Conv(ch, ch, activation=torch.nn.Identity())

    def forward(self, x):
        b, c, h, w = x.shape

        qkv = self.qkv(x)
        qkv = qkv.view(b, self.num_head, self.dim_key * 2 + self.dim_head, h * w)

        q, k, v = qkv.split([self.dim_key, self.dim_key, self.dim_head], dim=2)

        attn = (q.transpose(-2, -1) @ k) * self.scale
        attn = attn.softmax(dim=-1)

        x = (v @ attn.transpose(-2, -1)).view(b, c, h, w) + self.conv1(v.reshape(b, c, h, w))
        return self.conv2(x)


class PSABlock(torch.nn.Module):
    def __init__(self, ch, num_head):
        super().__init__()
        self.conv1 = Attention(ch, num_head)
        self.conv2 = torch.nn.Sequential(
            Conv(ch, ch * 2, activation=torch.nn.SiLU()),
            Conv(ch * 2, ch, activation=torch.nn.Identity())
        )

    def forward(self, x):
        x = x + self.conv1(x)
        return x + self.conv2(x)


class PSA(torch.nn.Module):

    def __init__(self, ch, n):
        super().__init__()
        self.conv1 = Conv(ch, 2 * (ch // 2), activation=torch.nn.SiLU())
        self.conv2 = Conv(2 * (ch // 2), ch, activation=torch.nn.SiLU())
        self.res_m = torch.nn.Sequential(*(PSABlock(ch // 2, ch // 128) for _ in range(n)))

    def forward(self, x):
        x, y = self.conv1(x).chunk(2, 1)
        return self.conv2(torch.cat(tensors=(x, self.res_m(y)), dim=1))


class DFL(torch.nn.Module):
    # Generalized Focal Loss
    # https://ieeexplore.ieee.org/document/9792391
    def __init__(self, ch=16):
        super().__init__()
        self.ch = ch
        self.conv = torch.nn.Conv2d(ch, out_channels=1, kernel_size=1, bias=False).requires_grad_(False)
        x = torch.arange(ch, dtype=torch.float).view(1, ch, 1, 1)
        self.conv.weight.data[:] = torch.nn.Parameter(x)

    def forward(self, x):
        b, c, a = x.shape
        x = x.view(b, 4, self.ch, a).transpose(2, 1)
        return self.conv(x.softmax(1)).view(b, 4, a)


class C3K(torch.nn.Module):
    def __init__(self, in_ch, out_ch):
        super().__init__()
        self.conv1 = Conv(in_ch, out_ch // 2, activation=torch.nn.SiLU())
        self.conv2 = Conv(in_ch, out_ch // 2, activation=torch.nn.SiLU())
        self.conv3 = Conv(2 * (out_ch // 2), out_ch, activation=torch.nn.SiLU())
        self.res_m = torch.nn.Sequential(
            Residual(out_ch // 2, e=1.0),
            Residual(out_ch // 2, e=1.0)
        )

    def forward(self, x):
        y = self.res_m(self.conv1(x))  # Process half of the input channels
        # Process the other half directly, Concatenate along the channel dimension
        return self.conv3(torch.cat((y, self.conv2(x)), dim=1))


class C3K2(torch.nn.Module):
    def __init__(self, in_ch, out_ch, n, csp, r):
        super().__init__()
        self.conv1 = Conv(in_ch, 2 * (out_ch // r), activation=torch.nn.SiLU())
        self.conv2 = Conv((2 + n) * (out_ch // r), out_ch, activation=torch.nn.SiLU())

        if not csp:
            self.res_m = torch.nn.ModuleList(Residual(out_ch // r) for _ in range(n))
        else:
            self.res_m = torch.nn.ModuleList(C3K(out_ch // r, out_ch // r) for _ in range(n))

    def forward(self, x):
        y = list(self.conv1(x).chunk(2, 1))
        y.extend(m(y[-1]) for m in self.res_m)
        return self.conv2(torch.cat(y, dim=1))


"""
AutoSpeed Context block
"""


# class ASC(torch.nn.Module):
#     # def __init__(self, H, W, C):
#     def __init__(self, in_ch, out_ch, n, csp, r):
#         super(ASC, self).__init__()
#
#         # Standard
#         self.in_ch = in_ch
#         self.GeLU = nn.GELU()
#
#         # Context - Expansion Layers
#         # self.expand_layer_0 = nn.Conv1d(in_ch, self.H*self.W, 1, 1, 1)
#         # self.expand_layer_0 = nn.Linear(in_ch, 160 * 160)
#
#         # Context - Extraction Layers
#         self.context_layer_0 = nn.Conv2d(1, in_ch // r, 3, 1, 1)
#         self.context_layer_1 = nn.Conv2d(in_ch // r, in_ch, 3, 1, 1)
#         self.context_layer_2 = nn.Conv2d(in_ch, out_ch, 3, 1, 1)
#
#     def forward(self, x):
#         b, c, h, w = x.size()
#         # Pooling and averaging channel layers to get a single vector
#         y = torch.mean(x, dim=[2, 3])
#
#         # Expansion
#         # device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
#         expand_layer_0 = nn.Linear(self.in_ch, h * w, device=x.device, dtype=x.dtype).to(x.device)
#         c0 = expand_layer_0(y)
#         c0 = self.GeLU(c0)
#         c1 = c0.view(b, h, w)
#         c1 = self.GeLU(c1)
#
#         # Context
#         c2 = self.context_layer_0(c1.unsqueeze(1))
#         # c2 = self.context_layer_0(c1)
#         c2 = self.GeLU(c2)
#         c3 = self.context_layer_1(c2)
#         c4 = self.GeLU(c3)
#         # Attention
#         c4 = c4 * x + x
#         context = self.GeLU(c4)
#         context = self.context_layer_2(context)
#         # context = self.GeLU(context)
#
#         return context


class ASC(torch.nn.Module):
    def __init__(self, in_ch, out_ch, n, csp, r, h, w):
        super(ASC, self).__init__()

        # Standard
        self.in_ch = in_ch
        self.h = h
        self.w = w
        self.GeLU = nn.GELU()

        # Context - Expansion Layers
        self.exp0 = nn.Conv1d(self.in_ch, self.h * self.w, 3, 1, 1)

        # Context - Extraction Layers
        self.ctx0 = nn.Conv2d(1, in_ch // r, 3, 1, 1)
        self.ctx1 = nn.Conv2d(in_ch // r, in_ch, 3, 1, 1)
        self.ctx2 = nn.Conv2d(in_ch, out_ch, 3, 1, 1)

    def forward(self, x):
        b, c, h, w = x.size()

        # Pooling and averaging channel layers to get a single vector
        y = torch.mean(x, dim=[2, 3], keepdim=True)

        # Expansion
        # c0 = self.exp0(y.unsqueeze(2))
        c0 = self.exp0(y.squeeze(-1))
        c0 = self.GeLU(c0)
        c1 = c0.view(b, 1, self.h, self.w)
        c1 = self.GeLU(c1)

        # Context
        c2 = self.ctx0(c1)
        c2 = self.GeLU(c2)
        c3 = self.ctx1(c2)
        c4 = self.GeLU(c3)

        # Attention
        c4 = c4 * x + x

        context = self.GeLU(c4)
        context = self.ctx2(context)

        return context


class SPPF(nn.Module):

    def __init__(self, c1, c2, k=5):
        super().__init__()
        c_ = c1 // 2
        self.cv1 = Conv(c1, c_, activation=torch.nn.SiLU(), k=1)
        self.cv2 = Conv(c_ * 4, c2, activation=torch.nn.SiLU(), k=1)
        self.middle_block = nn.MaxPool2d(kernel_size=k, stride=1, padding=k // 2)

    def forward(self, x):
        x = self.cv1(x)
        y1 = self.middle_block(x)
        y2 = self.middle_block(y1)
        return self.cv2(torch.cat((x, y1, y2, self.middle_block(y2)), 1))  # Ending with Conv Block


class C2PSA(nn.Module):
    def __init__(self, c1, c2, e=0.5):
        super().__init__()
        self.c_ = int(c1 * e)
        self.cv1 = Conv(c1, 2 * self.c_, activation=torch.nn.SiLU(), k=1, s=1)
        self.cv2 = Conv(2 * self.c_, c2, activation=torch.nn.SiLU(), k=1, s=1)

        self.middle_block = PSABlock(self.c_, num_head=self.c_ // 64)

    def forward(self, x: torch.Tensor):
        x, y = self.cv1(x).split((self.c_, self.c_), dim=1)
        y = self.middle_block(y)
        return self.cv2(torch.cat((x, y), 1))
