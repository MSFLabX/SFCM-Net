import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.fft as fft


class CNN_Encoder(nn.Module):
    def __init__(self, l1, l2):
        super(CNN_Encoder, self).__init__()
        self.conv1 = nn.Sequential(
            nn.Conv2d(l1, 32, 3, 1, 1),
            nn.BatchNorm2d(32),
            nn.LeakyReLU(),
        )
        self.conv2 = nn.Sequential(
            nn.Conv2d(l2, 32, 3, 1, 1),
            nn.BatchNorm2d(32),
            nn.LeakyReLU(),
        )

    def forward(self, x11, x21, x12, x22, x13, x23):
        x11 = self.conv1(x11)
        x12 = self.conv1(x12)
        x13 = self.conv1(x13)

        x21 = self.conv2(x21)
        x22 = self.conv2(x22)
        x23 = self.conv2(x23)

        return x11, x21, x12, x22, x13, x23


class CNN_Classifier(nn.Module):
    def __init__(self, Classes):
        super(CNN_Classifier, self).__init__()
        self.conv1 = nn.Sequential(
            nn.Conv2d(64, 32, 1),
            nn.BatchNorm2d(32),
            nn.LeakyReLU(),
            nn.AdaptiveAvgPool2d(1),
        )
        self.conv2 = nn.Sequential(
            nn.Conv2d(32, Classes, 1),
        )

    def forward(self, x):
        x = self.conv1(x)
        x = self.conv2(x)
        x = x.view(x.size(0), -1)
        x_out = F.softmax(x, dim=1)
        return x_out


class CNL(nn.Module):
    def __init__(self, high_dim, low_dim, flag=0):
        super(CNL, self).__init__()
        self.high_dim = high_dim
        self.low_dim = low_dim

        self.g = nn.Conv2d(self.low_dim, self.low_dim, kernel_size=1, stride=1, padding=0)
        self.theta = nn.Conv2d(self.high_dim, self.low_dim, kernel_size=1, stride=1, padding=0)
        self.phi = nn.Conv2d(self.low_dim, self.low_dim, kernel_size=1, stride=1, padding=0)
        self.W = nn.Sequential(
            nn.Conv2d(self.low_dim, self.high_dim, kernel_size=1, stride=1, padding=0),
            nn.BatchNorm2d(high_dim),
        )

        self.W_h = nn.Conv2d(self.low_dim, self.low_dim, kernel_size=1, stride=1, padding=0)
        nn.init.constant_(self.W[1].weight, 0.0)
        nn.init.constant_(self.W[1].bias, 0.0)

    def forward(self, x_h, x_l):
        B = x_h.size(0)
        g_x = self.g(x_l).view(B, self.low_dim, -1)

        theta_x = self.theta(x_h).view(B, self.low_dim, -1)
        phi_x = self.phi(x_l).view(B, self.low_dim, -1).permute(0, 2, 1)

        energy = torch.matmul(theta_x, phi_x)
        attention = energy / energy.size(-1)

        y = torch.matmul(attention, g_x)
        y = y.view(B, self.low_dim, *x_l.size()[2:])
        W_y = self.W(y)
        z = W_y + self.W_h(x_h)

        return z


class PNL(nn.Module):
    def __init__(self, high_dim, low_dim, reduc_ratio=1):
        super(PNL, self).__init__()
        self.high_dim = high_dim
        self.low_dim = low_dim
        self.reduc_ratio = reduc_ratio

        self.g = nn.Conv2d(self.low_dim, self.low_dim // self.reduc_ratio, kernel_size=1, stride=1, padding=0)
        self.theta = nn.Conv2d(self.high_dim, self.low_dim // self.reduc_ratio, kernel_size=1, stride=1, padding=0)
        self.phi = nn.Conv2d(self.low_dim, self.low_dim // self.reduc_ratio, kernel_size=1, stride=1, padding=0)

        self.W = nn.Sequential(
            nn.Conv2d(self.low_dim // self.reduc_ratio, self.high_dim, kernel_size=1, stride=1, padding=0),
            nn.BatchNorm2d(high_dim),
        )
        self.W_h = nn.Conv2d(self.low_dim, self.low_dim, kernel_size=1, stride=1, padding=0)
        nn.init.constant_(self.W[1].weight, 0.0)
        nn.init.constant_(self.W[1].bias, 0.0)

    def forward(self, x_h, x_l):
        B = x_h.size(0)
        g_x = self.g(x_l).view(B, self.low_dim, -1)
        g_x = g_x.permute(0, 2, 1)

        theta_x = self.theta(x_h).view(B, self.low_dim, -1)
        theta_x = theta_x.permute(0, 2, 1)

        phi_x = self.phi(x_l).view(B, self.low_dim, -1)

        energy = torch.matmul(theta_x, phi_x)
        attention = energy / energy.size(-1)

        y = torch.matmul(attention, g_x)
        y = y.permute(0, 2, 1).contiguous()
        y = y.view(B, self.low_dim // self.reduc_ratio, *x_h.size()[2:])
        W_y = self.W(y)
        z = W_y + self.W_h(x_h)
        return z


class APCI(nn.Module):
    def __init__(self, high_dim, low_dim):
        super(APCI, self).__init__()
        self.CNL = CNL(high_dim, low_dim)
        self.PNL = PNL(high_dim, low_dim)

    def forward(self, x, x0):
        z = self.CNL(x, x0)
        z = self.PNL(z, x0)
        return z


class MFA_SFF(nn.Module):

    def __init__(self, in_channels=64):
        super().__init__()
        self.conv1 = nn.Sequential(
            nn.Conv2d(64, 64, 3, 1, 1),
            nn.BatchNorm2d(64),
            nn.LeakyReLU(),
            nn.MaxPool2d(2),
        )

        self.conv2 = nn.Sequential(
            nn.Conv2d(64, 64, 3, 1, 1),
            nn.BatchNorm2d(64),
            nn.LeakyReLU(),
            nn.MaxPool2d(2),
        )

        self.APCI_A = APCI(high_dim=in_channels, low_dim=in_channels)
        self.APCI_P = APCI(high_dim=in_channels, low_dim=in_channels)

        self.conv_A = nn.Sequential(
            nn.Conv2d(in_channels, in_channels, kernel_size=1, padding=0),
            nn.ReLU(inplace=True)
        )
        self.conv_P = nn.Sequential(
            nn.Conv2d(in_channels, in_channels, kernel_size=1, padding=0),
            nn.ReLU(inplace=True)
        )

    def forward(self, F_tar, F_ref):
        F_tar = self.conv1(F_tar)
        F_ref = self.conv2(F_ref)
        F_tar_fft = fft.fft2(F_tar, dim=(-2, -1))
        F_ref_fft = fft.fft2(F_ref, dim=(-2, -1))

        A_tar = torch.abs(F_tar_fft)
        P_tar = torch.angle(F_tar_fft)
        A_ref = torch.abs(F_ref_fft)
        P_ref = torch.angle(F_ref_fft)

        A_fused = A_tar + A_ref
        P_fused = P_tar + P_ref

        A_enhanced_mfa = self.APCI_A(A_fused, P_fused)
        P_enhanced_mfa = self.APCI_P(P_fused, A_fused)

        A_enhanced = self.conv_A(A_enhanced_mfa)
        P_enhanced = self.conv_P(P_enhanced_mfa)

        F_fre_fft = A_enhanced * torch.exp(1j * P_enhanced)
        F_fre = fft.ifft2(F_fre_fft, dim=(-2, -1)).real

        return F_fre


class CB(nn.Module):
    def __init__(self,l):
        super().__init__()
        self.l = l
        self.norm1 = nn.LayerNorm(l)
        self.qkv_proj = nn.Conv2d(l, l * 3, kernel_size=1)
        self.attn_proj = nn.Conv2d(l, l, kernel_size=1)
        self.point_conv1 = nn.Conv2d(l, 64, kernel_size=1)
        self.norm2 = nn.LayerNorm(64)
        self.point_conv2 = nn.Conv2d(64, 64, kernel_size=1)
        self.global_pool = nn.AdaptiveAvgPool2d(1)

    def forward(self, x):
        residual = x
        x_norm = self.norm1(x.permute(0, 2, 3, 1)).permute(0, 3, 1, 2)
        B, C, H, W = x_norm.shape
        qkv = self.qkv_proj(x_norm)
        q, k, v = qkv.chunk(3, dim=1)
        q = q.flatten(2)
        k = k.flatten(2)
        v = v.flatten(2)
        attn = F.softmax(torch.matmul(q.transpose(1, 2), k) / (self.l ** 0.5), dim=-1)
        attn_out = torch.matmul(v, attn.transpose(1, 2)).view(B, C, H, W)
        attn_out = self.attn_proj(attn_out)
        x = residual + attn_out
        x = self.point_conv1(x)
        x_norm2 = self.norm2(x.permute(0, 2, 3, 1)).permute(0, 3, 1, 2)
        x = self.point_conv2(x_norm2)
        x = self.global_pool(x)
        return x


class LayerNorm2d(nn.Module):
    def __init__(self, channels, eps=1e-6):
        super().__init__()
        self.eps = eps
        self.weight = nn.Parameter(torch.ones(channels))
        self.bias = nn.Parameter(torch.zeros(channels))
        self.spatial_scale = nn.Parameter(torch.ones(1, channels, 1, 1))

    def forward(self, x):
        mu = x.mean(dim=1, keepdim=True)
        var = (x - mu).pow(2).mean(dim=1, keepdim=True)
        x_hat = (x - mu) / torch.sqrt(var + self.eps)
        return self.spatial_scale * x_hat * self.weight.view(1, -1, 1, 1) + self.bias.view(1, -1, 1, 1)


class SimpleGate(nn.Module):
    def forward(self, x):
        x1, x2 = x.chunk(2, dim=1)
        return x1 * x2


class FreMLPPlus(nn.Module):
    def __init__(self, nc, expand=2):
        super().__init__()
        hidden = max(1, expand * nc)

        self.mag_mlp = nn.Sequential(
            nn.Conv2d(nc, hidden, 1),
            nn.GELU(),
            nn.Conv2d(hidden, nc, 1)
        )

        se_hidden = max(1, nc // 4)
        self.freq_se = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(nc, se_hidden, 1),
            nn.GELU(),
            nn.Conv2d(se_hidden, nc, 1),
            nn.Sigmoid()
        )
        self.phase_smooth = nn.Conv2d(nc, nc, 1, bias=False)

    def forward(self, x):
        B, C, H, W = x.shape

        x_freq = torch.fft.rfft2(x, norm='ortho')
        mag = torch.abs(x_freq)
        pha = torch.angle(x_freq)

        mag = mag.clamp_min(1e-6)
        mag = self.mag_mlp(mag)
        att = self.freq_se(mag)
        mag = mag * att
        pha = self.phase_smooth(pha)
        real = mag * torch.cos(pha)
        imag = mag * torch.sin(pha)
        x_complex = torch.complex(real, imag)
        out = torch.fft.irfft2(x_complex, s=(H, W), norm='ortho')
        return out


class Branch(nn.Module):
    def __init__(self, c, expands, dilation):
        super().__init__()
        self.dw = expands * c
        self.branch = nn.Conv2d(
            in_channels=self.dw,
            out_channels=self.dw,
            kernel_size=3,
            stride=1,
            padding=dilation,
            dilation=dilation,
            groups=self.dw,
            bias=True
        )

    def forward(self, x):
        return self.branch(x)


class SFCE(nn.Module):
    def __init__(self, c, DW_Expand=2, dilations=(1, 2)):
        super().__init__()
        self.c = c
        self.dw_channel = DW_Expand * c
        self.conv1 = nn.Conv2d(c, self.dw_channel, kernel_size=1, bias=True)
        self.branches = nn.ModuleList([Branch(c, DW_Expand, d) for d in dilations])
        self.branch_weight = nn.Parameter(torch.ones(len(dilations)))
        self.sca = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(self.dw_channel // 2, self.dw_channel // 2, kernel_size=1, bias=True),
            nn.GELU()
        )
        self.conv3 = nn.Conv2d(self.dw_channel // 2, c, kernel_size=1, bias=True)
        self.norm1 = LayerNorm2d(c)
        self.norm2 = LayerNorm2d(c)
        self.freq = FreMLPPlus(nc=c, expand=2)
        self.cci = nn.Sequential(
            nn.Conv2d(c, c, 1, bias=True),
            nn.GELU(),
            nn.Conv2d(c, c, 1, bias=True)
        )
        self.gamma = nn.Parameter(torch.zeros(1, c, 1, 1))
        self.beta = nn.Parameter(torch.zeros(1, c, 1, 1))
        self.sg = SimpleGate()
        self.conv4 = nn.Sequential(
            nn.Conv2d(32, 64, 3, 1, 1),
            nn.BatchNorm2d(64),
            nn.LeakyReLU(),
            nn.MaxPool2d(2),
            nn.Dropout(0.4),
        )

    def forward(self, inp):
        x = self.norm1(inp)
        x = self.conv1(x)
        weights = torch.softmax(self.branch_weight, dim=0)
        z = 0
        for w, branch in zip(weights, self.branches):
            z = z + w * branch(x)
        z = self.sg(z)
        x_spa = self.sca(z) * z
        x_spa = self.conv3(x_spa)
        y = inp + self.beta * x_spa
        x2 = self.norm2(y)
        x_freq = self.freq(x2)
        x_freq = self.cci(x_freq)
        out = y + self.gamma * (y * x_freq)
        out = self.conv4(out)
        return out


class SFCM(nn.Module):
    def __init__(self, l1, l2, num_classes):
        super().__init__()
        self.cnn_encoder = CNN_Encoder(l1, l2)

        dim = 64
        self.SFCE11 = SFCE(32, DW_Expand=2, dilations=(1, 2))
        self.SFCE12 = SFCE(32, DW_Expand=2, dilations=(1, 2))
        self.SFCE13 = SFCE(32, DW_Expand=2, dilations=(1, 2))
        self.SFCE21 = SFCE(32, DW_Expand=2, dilations=(1, 2))
        self.SFCE22 = SFCE(32, DW_Expand=2, dilations=(1, 2))
        self.SFCE23 = SFCE(32, DW_Expand=2, dilations=(1, 2))
        self.mfa1 = MFA_SFF(dim)
        self.mfa2 = MFA_SFF(dim)
        self.mfa3 = MFA_SFF(dim)

        self.CB = CB(l1)
        self.cnn_classifier1 = CNN_Classifier(num_classes)
        self.cnn_classifier2 = CNN_Classifier(num_classes)
        self.cnn_classifier3 = CNN_Classifier(num_classes)

    def encoder(self, x11, x21, x12, x22, x13, x23):
        x1_1, x2_1, x1_2, x2_2, x1_3, x2_3, = self.cnn_encoder(x11, x21, x12, x22, x13, x23)

        return x1_1, x2_1, x1_2, x2_2, x1_3, x2_3

    def classifier(self, x1, x2, x3):
        x_cls1 = self.cnn_classifier1(x1)
        x_cls2 = self.cnn_classifier2(x2)
        x_cls3 = self.cnn_classifier3(x3)
        x_cls = x_cls1 + x_cls2 + x_cls3
        return x_cls

    def forward(self, img11, img21, img12, img22, img13, img23):
        x1_1, x2_1, x1_2, x2_2, x1_3, x2_3 = self.encoder(img11, img21, img12, img22, img13, img23)
        x1_1 = self.SFCE11(x1_1)
        x1_2 = self.SFCE12(x1_2)
        x1_3 = self.SFCE13(x1_3)
        x2_1 = self.SFCE21(x2_1)
        x2_2 = self.SFCE22(x2_2)
        x2_3 = self.SFCE23(x2_3)

        center_block = img11[:, :, 2:4, 2:4]
        xc = self.CB(center_block)
        x1 = self.mfa1(x1_1, x2_1)
        x2 = self.mfa2(x1_2, x2_2)
        x3 = self.mfa3(x1_3, x2_3)

        x_cls = self.classifier(x1*xc, x2*xc, x3*xc)

        return x_cls



