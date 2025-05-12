import torch
import torch.nn as nn


        

class SelfAttentionBlock(nn.Module):
    def __init__(self, in_channels):
        super(SelfAttentionBlock, self).__init__()
        self.in_channels = in_channels

        self.group_norm = nn.GroupNorm(32, in_channels, eps=1e-6)
        self.multihead_attention = nn.MultiheadAttention(in_channels, num_heads=4, batch_first=True)
    
    def forward(self, x, *args):
        B, _, H, W = x.shape
        h = self.group_norm(x)
        h = h.reshape(B, self.in_channels, H * W).swapaxes(1, 2)
        h, _ = self.multihead_attention(h, h, h, need_weights=False)
        h = h.swapaxes(2, 1).reshape(B, self.in_channels, H, W)
        return x + h




class UpSampleBlock(nn.Module):
    def __init__(self, in_channels, use_conv=False):
        super(UpSampleBlock, self).__init__()

        self.use_conv = use_conv

        if self.use_conv:
            self.conv = nn.Conv2d(in_channels, in_channels, kernel_size=3, stride=1, padding=1)

    def forward(self, x, *args):
        x = nn.functional.interpolate(x, scale_factor=2, mode='nearest')
        if self.use_conv:
            x = self.conv(x)
        return x

class DownSampleBlock(nn.Module):
    def __init__(self, in_channels):
        super(DownSampleBlock, self).__init__()

        self.conv = nn.Conv2d(in_channels, in_channels, kernel_size=3, stride=2, padding=1)

    def forward(self, x, *args):
        return self.conv(x)


class ResidualBlock(nn.Module):
    def __init__(self, in_channels, out_channels, time_emb_dim, apply_attention):
        super(ResidualBlock, self).__init__()

        self.in_channels = in_channels
        self.out_channels = out_channels
        self.time_emb_dim = time_emb_dim
        self.apply_attention = apply_attention

        self.resnet_block = nn.ModuleList()

        self.group_norm1 = nn.GroupNorm(32, in_channels, eps=1e-6)
        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=1, padding=1)
        self.group_norm2 = nn.GroupNorm(32, out_channels, eps=1e-6)
        self.conv2 = nn.Conv2d(out_channels, out_channels, kernel_size=3, stride=1, padding=1)
        self.act = nn.SiLU()
        self.dropout = nn.Dropout(0.0)

        if self.in_channels == self.out_channels:
            self.skip_connection = nn.Identity()
        else:
            self.skip_connection = nn.Conv2d(in_channels, out_channels, kernel_size=1, stride=1)


    def forward(self, x, t):
        h = x
        h = self.group_norm1(h)
        h = self.act(h)
        h = self.conv1(h)

        h = self.group_norm2(h)
        h = self.act(h)
        h = self.dropout(h)
        h = self.conv2(h)

        h = h + self.skip_connection(x)

        return h


class Encoder(nn.Module):
    def __init__(self, in_channels, hidden_dim, num_downsample_blocks, num_bottleneck_blocks):

        super(Encoder, self).__init__()
        self.in_channels = in_channels
        self.num_downsample_blocks = num_downsample_blocks
        self.hidden_dim = hidden_dim

        self.first_layer = nn.Conv2d(in_channels, hidden_dim, kernel_size=3, stride=1, padding=1)

        self.encoder = nn.ModuleList()

        self.downsample_blocks = nn.ModuleList()

        in_channels = hidden_dim
        for num_block in range(num_downsample_blocks):
            for num_resnet_block in range(num_bottleneck_blocks-1):
                resnet_block = ResidualBlock(in_channels=in_channels, out_channels=hidden_dim, time_emb_dim=None, apply_attention=False)
                in_channels = hidden_dim
                self.downsample_blocks.append(resnet_block)

            if num_block != num_downsample_blocks - 1:
                downsample_block = DownSampleBlock(hidden_dim)
                self.downsample_blocks.append(downsample_block)
                in_channels = hidden_dim
                if num_block != num_downsample_blocks - 2:
                    hidden_dim *= 2

        self.bottleneck_blocks = nn.ModuleList()

        self.bottleneck_blocks.append(ResidualBlock(hidden_dim, hidden_dim, None, False))
        self.bottleneck_blocks.append(ResidualBlock(hidden_dim, hidden_dim, None, False))
        self.bottleneck_blocks.append(SelfAttentionBlock(hidden_dim))

        #self.last_layer = nn.Conv2d(hidden_dim, 8, kernel_size=3, stride=1, padding=1)
        self.pool = nn.AdaptiveAvgPool2d((4, 4))
        self.last_conv = nn.Conv2d(hidden_dim, 8, kernel_size=1)
    
    def forward(self, x, t):

        h = self.first_layer(x)
            
        for layer in self.downsample_blocks:
            h = layer(h, None)

        for layer in self.bottleneck_blocks:
            h = layer(h, None)

        #h = self.pool(h)
        h = self.last_conv(h)

        #h = self.last_layer(h)
        return h
    
class Decoder(nn.Module):
    def __init__(self, in_channels, out_channels, hidden_dim, num_upsample_blocks, num_resnet_blocks):
        super(Decoder, self).__init__()

        self.upsample_blocks = nn.ModuleList()

        self.first_layer = nn.Conv2d(in_channels, hidden_dim, kernel_size=3, stride=1, padding=1)

         # Projection to upsample from 4x4 or 8x8 to 64x64
        # self.project_up = nn.Sequential(
        #     nn.ConvTranspose2d(in_channels, hidden_dim, kernel_size=2, stride=2),  # from (B, in_channels, 4x4) → (B, hidden_dim, 16x16)
        #     nn.GroupNorm(32, hidden_dim),
        #     nn.SiLU(),
        #     nn.ConvTranspose2d(hidden_dim, hidden_dim, kernel_size=4, stride=4),  # from (B, hidden_dim, 16x16) → (B, hidden_dim, 64x64)
        # )

        in_channels = hidden_dim
        for num_block in range(num_upsample_blocks):
            for _ in range(num_resnet_blocks-1):
                resnet_block = ResidualBlock(in_channels, hidden_dim, None, False)
                self.upsample_blocks.append(resnet_block)
                in_channels = hidden_dim

            if num_block != num_upsample_blocks - 1:
                upsample_block = UpSampleBlock(hidden_dim, use_conv=True)
                self.upsample_blocks.append(upsample_block)

            if num_block >= 1:
                in_channels = hidden_dim
                hidden_dim //= 2

        self.last_layer = nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=1, padding=1)


    def forward(self, x, *args):
        h = self.first_layer(x)

        #h = x
        #h = self.project_up(x)
        
        for layer in self.upsample_blocks:
            h = layer(h, None)

        h = self.last_layer(h)

        return h
    
class VAE(nn.Module):
    def __init__(self):
        super(VAE, self).__init__()

        self.encoder = Encoder(in_channels=3, hidden_dim=64, num_downsample_blocks=3, num_bottleneck_blocks=2)
        self.decoder = Decoder(in_channels=4, out_channels=3, hidden_dim=256, num_upsample_blocks=3, num_resnet_blocks=2)
    
    def forward(self, x):
        # Add forward pass logic here
        h = self.encoder(x, None)
        mu, logvar = h.chunk(2, dim=1)
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        z = mu + eps * std
        z = z.to("cuda")

        h_dec = self.decoder(z, None)
        return h_dec, mu, logvar

        
        


# fake_image = torch.randn(1, 3, 256, 256)

# fake_latent = torch.randn(1, 4, 32, 32)

# ###decoder = Decoder(in_channels=4, out_channels=3, hidden_dim=512, num_upsample_blocks=4, num_resnet_blocks=3)
# #h = decoder(fake_latent, None)


# encoder = Encoder(in_channels=3, hidden_dim=128, num_downsample_blocks=4, num_bottleneck_blocks=2)
# h = encoder(fake_image, None)
