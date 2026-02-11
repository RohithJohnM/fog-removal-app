import torch
import torch.nn as nn

# --- Helper: Standard Convolution ---
def default_conv(in_channels, out_channels, kernel_size, bias=True):
    return nn.Conv2d(in_channels, out_channels, kernel_size, padding=(kernel_size//2), bias=bias)

# --- Part 1: Pixel Attention (PA) ---
class PALayer(nn.Module):
    def __init__(self, channel):
        super(PALayer, self).__init__()
        self.pa = nn.Sequential(
            nn.Conv2d(channel, channel // 8, 1, padding=0, bias=True),
            nn.ReLU(inplace=True),
            nn.Conv2d(channel // 8, 1, 1, padding=0, bias=True),
            nn.Sigmoid()
        )
    def forward(self, x):
        y = self.pa(x)
        return x * y

# --- Part 2: Channel Attention (CA) ---
class CALayer(nn.Module):
    def __init__(self, channel):
        super(CALayer, self).__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.ca = nn.Sequential(
            nn.Conv2d(channel, channel // 8, 1, padding=0, bias=True),
            nn.ReLU(inplace=True),
            nn.Conv2d(channel // 8, channel, 1, padding=0, bias=True),
            nn.Sigmoid()
        )
    def forward(self, x):
        y = self.avg_pool(x)
        y = self.ca(y)
        return x * y

# --- Part 3: The Basic Block ---
class Block(nn.Module):
    def __init__(self, conv, dim, kernel_size):
        super(Block, self).__init__()
        self.conv1 = conv(dim, dim, kernel_size)
        self.act1 = nn.ReLU(inplace=True)
        self.conv2 = conv(dim, dim, kernel_size)
        self.calayer = CALayer(dim)
        self.palayer = PALayer(dim)
    def forward(self, x):
        res = self.act1(self.conv1(x))
        res = res + x 
        res = self.conv2(res)
        res = self.calayer(res)
        res = self.palayer(res)
        res += x 
        return res

# --- Part 4: The Group Architecture ---
class Group(nn.Module):
    def __init__(self, conv, dim, kernel_size, blocks):
        super(Group, self).__init__()
        modules = [Block(conv, dim, kernel_size) for _ in range(blocks)]
        self.gp = nn.Sequential(*modules)
        self.conv = conv(dim, dim, kernel_size)
    def forward(self, x):
        res = self.gp(x)
        res = self.conv(res)
        return res + x

# --- Part 5: The Full FFA-Net Model ---
class FFA(nn.Module):
    def __init__(self, gps=3, blocks=19):
        super(FFA, self).__init__()
        self.gps = gps
        self.dim = 64
        kernel_size = 3
        
        # Pre-processing
        self.pre = default_conv(3, self.dim, kernel_size)
        
        # Backbone (The heavy lifting)
        self.g_layers = nn.ModuleList([
            Group(default_conv, self.dim, kernel_size, blocks=blocks) for _ in range(gps)
        ])
        
        # Post-processing (reconstructing the image)
        self.post = nn.Sequential(
            default_conv(self.dim, self.dim, kernel_size),
            default_conv(self.dim, 3, kernel_size)
        )

    def forward(self, x):
        x1 = x                  # Save original image
        x = self.pre(x)         # Feature extraction
        res = x
        for i in range(self.gps):
            res = self.g_layers[i](res)
        res = self.post(res)    # Convert back to image
        return res + x1         # Add to ORIGINAL image (x1), not features (x)