import torch
import torch.nn as nn
import math

class SinusoidalTimeEmbeddings(nn.Module):
    def __init__(self, num_steps, time_emb_dim, hidden_dim):
        super(SinusoidalTimeEmbeddings, self).__init__()
        self.time_emb_dim = time_emb_dim
        self.num_steps = num_steps

        half_dim = time_emb_dim // 2
        emb = math.log(1000) / (half_dim - 1) 
        emb = torch.exp(torch.arange(half_dim) * -emb).unsqueeze(0)

        ts = torch.arange(num_steps).unsqueeze(-1)
        emb = ts * emb

        emb = torch.cat((emb.sin(), emb.cos()), dim=-1)

        self.time_block = nn.Sequential(
            nn.Embedding.from_pretrained(emb),
            nn.Linear(time_emb_dim, hidden_dim),
            nn.SiLU(),
            nn.Linear(hidden_dim, hidden_dim),
        )

    def forward(self, t):
        return self.time_block(t)

class AttentionBlock(nn.Module):
    def __init__(self, channels=64):
        super(AttentionBlock, self).__init__()
        self.channels = channels

        self.group_norm = nn.GroupNorm(8, channels)
        self.multihead_attn = nn.MultiheadAttention(channels, num_heads=4, batch_first=True)

    def forward(self, x):
        B, _, H, W = x.shape
        h = self.group_norm(x)
        h = h.reshape(B, self.channels, H *W).swapaxes(1, 2)
        h, _ = self.multihead_attn(h, h, h)
        h = h.swapaxes(2, 1).reshape(B, self.channels, H, W)
        return x + h

class ResidualBlock(nn.Module):
    def __init__(self, in_channels, out_channels, time_emb_dim, apply_attention, dropout_rate = 0.1):
        super(ResidualBlock, self).__init__()

        self.group_norm1 = nn.GroupNorm(8, in_channels)
        self.act = nn.SiLU()
        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=1, padding="same")

        self.dropout = nn.Dropout(dropout_rate)

        self.time_emb_matching = nn.Linear(time_emb_dim, out_channels)

        self.group_norm2 = nn.GroupNorm(8, out_channels)
        self.conv2 = nn.Conv2d(out_channels, out_channels, kernel_size=3, stride=1, padding="same")

        if in_channels != out_channels:
            self.skip_connection = nn.Conv2d(in_channels, out_channels, kernel_size=1, stride=1)
        else:
            self.skip_connection = nn.Identity()

        if apply_attention:
            self.attention = AttentionBlock(out_channels)
        else:
            self.attention = nn.Identity()
        
    def forward(self, x, t):
        h = self.group_norm1(x)
        h = self.act(h)
        h = self.conv1(h)
        t = self.time_emb_matching(self.act(t))
        h += t[:,:, None, None]

        h = self.group_norm2(h)
        h = self.act(h)
        h = self.dropout(h)
        h = self.conv2(h)

        h = h + self.skip_connection(x)

        h = self.attention(h)

        return h
    
class UNet(nn.Module):
    def __init__(self, in_channels, hidden_channels, time_emb_dim, num_blocks = 4, num_resnet_blocks=2, dropout_rate=0.1):
        super(UNet, self).__init__()
        self.num_resnet_blocks = num_resnet_blocks
        self.time_emb = SinusoidalTimeEmbeddings(num_steps=1000, time_emb_dim=64, hidden_dim=256)
        self.in_channels = in_channels

        self.first_layer = nn.Conv2d(1, 64, kernel_size=3, stride=1, padding="same")
        self.apply_attention = (False, True, True, False)

        self.downsample_blocks = nn.ModuleList()
        self.encoder_channels = [in_channels]
        self.in_channels_mult = (1, 2, 2, 1)
        self.decoder_channels_mult = (1, 2, 4, 4)

        self.class_embedding = nn.Embedding(10, time_emb_dim)

        for num_block in range(num_blocks):
            hidden_channels *= self.in_channels_mult[num_block]

            for i in range(self.num_resnet_blocks):
                resnet_block = ResidualBlock(in_channels,hidden_channels,time_emb_dim,self.apply_attention[num_block],dropout_rate)
                self.downsample_blocks.append(resnet_block)
                in_channels = hidden_channels
                self.encoder_channels.append(in_channels)

            if num_block != num_blocks - 1:
                self.downsample_blocks.append(DownSampleBlock(in_channels))
                self.encoder_channels.append(in_channels)

        self.bottleneck_blocks = nn.ModuleList()
        self.bottleneck_blocks.append(ResidualBlock(in_channels, in_channels, time_emb_dim, True, dropout_rate))
        self.bottleneck_blocks.append(ResidualBlock(in_channels, in_channels, time_emb_dim, False, dropout_rate))

        self.upsample_blocks = nn.ModuleList()

        in_channels = hidden_channels

        for num_block in reversed(range(num_blocks)):
            hidden_channels = self.in_channels * self.decoder_channels_mult[num_block]

            for i in range(self.num_resnet_blocks + 1):
                encoder_match_channels = self.encoder_channels.pop()
                resnet_block = ResidualBlock(in_channels + encoder_match_channels, hidden_channels, time_emb_dim, self.apply_attention[num_block], dropout_rate)
                self.upsample_blocks.append(resnet_block)
                in_channels = hidden_channels

            if num_block != 0:
                self.upsample_blocks.append(UpSampleBlock(in_channels))

        self.last_layer = nn.Sequential(
            nn.GroupNorm(8, in_channels),
            nn.SiLU(),
            nn.Conv2d(in_channels, 1, kernel_size=3, stride=1, padding="same"),
        )

    def forward(self, x, t, c):
        time_embeddings = self.time_emb(t)
        c = self.class_embedding(c)

        time_embeddings = time_embeddings + c



        h = self.first_layer(x)
        downsample_outs = [h]

        for layer in self.downsample_blocks:
            h = layer(h, time_embeddings)
            downsample_outs.append(h)

        for layer in self.bottleneck_blocks:
            h = layer(h, time_embeddings)

        shapes = [out.shape for out in downsample_outs]
        
        for layer in self.upsample_blocks:
            if isinstance(layer, ResidualBlock):
                downsample_out = downsample_outs.pop()
                h = torch.cat((h, downsample_out), dim=1)
            else:
                pass
            h = layer(h, time_embeddings)

        h = self.last_layer(h)
        return h


class UpSampleBlock(nn.Module):
    def __init__(self, in_channels):
        super().__init__()

        self.upsample = nn.Sequential(
            nn.Upsample(scale_factor=2, mode="nearest"),
            nn.Conv2d(in_channels=in_channels, out_channels=in_channels, kernel_size=3, stride=1, padding=1),
        )

    def forward(self, x, *args):
        return self.upsample(x)
    
class DownSampleBlock(nn.Module):
    def __init__(self, in_channels):
        super(DownSampleBlock, self).__init__()
        
        self.conv = nn.Conv2d(in_channels, in_channels, kernel_size=3, stride=2, padding=1)

    def forward(self, x, *args):
        x = self.conv(x)
        return x

