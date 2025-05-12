import torch

import torch.nn as nn
import math

class SinusoidalPositionEmbeddings(nn.Module):
    def __init__(self, timesteps = 1000, embedding_dim = 128, embedding_dim_exp = 512):
        super().__init__()

        half_dim = embedding_dim // 2

        emb = math.log(1000) / (half_dim - 1)
        emb = torch.exp(torch.arange(half_dim, dtype=torch.float32) * -emb).unsqueeze(0)

        ts = torch.arange(timesteps, dtype=torch.float32).unsqueeze(-1)

        emb = ts * emb

        emb = torch.cat((emb.sin(), emb.cos()), dim=-1)

        self.time_block = nn.Sequential(
            nn.Embedding.from_pretrained(emb),
            nn.Linear(embedding_dim, embedding_dim_exp),
            nn.SiLU(),
            nn.Linear(embedding_dim_exp, embedding_dim_exp),
        )

    def forward(self, time):
        return self.time_block(time)

class AttentionLayer(nn.Module):
    def __init__(self, channels=64):
        super().__init__()
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


class ResNetBlock(nn.Module):
    def __init__(self, *, in_channels, out_channels, dropout_rate=0.1, time_emb_dims = 512, apply_attention=False):
        super().__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels

        self.act_fn = nn.SiLU()

        self.normalize1 = nn.GroupNorm(8, self.in_channels)
        self.conv1 = nn.Conv2d(self.in_channels, self.out_channels, kernel_size=3, stride=1, padding="same")

        self.dense1 = nn.Linear(time_emb_dims, self.out_channels)

        self.normalize2 = nn.GroupNorm(8, self.out_channels)
        self.dropout = nn.Dropout2d(p=dropout_rate)
        self.conv2 = nn.Conv2d(self.out_channels, self.out_channels, kernel_size=3, stride=1, padding="same")

        if self.in_channels != self.out_channels:
            self.match_input = nn.Conv2d(self.in_channels, self.out_channels, kernel_size=1, stride=1)
        else:
            self.match_input = nn.Identity()

        if apply_attention:
            self.attention = AttentionLayer(self.out_channels)
        else:
            self.attention = nn.Identity()

    def forward(self, x, t):
        h = self.act_fn(self.normalize1(x))
        h = self.conv1(h)

        h+= self.dense1(self.act_fn(t))[: ,:, None, None]

        h = self.act_fn(self.normalize2(h))
        h = self.dropout(h)
        h = self.conv2(h)

        h = h + self.match_input(x)
        h = self.attention(h)

        return h

class DownSample(nn.Module):
    def __init__(self, in_channels):
        super().__init__()
        self.downsample = nn.Sequential(
            nn.Conv2d(in_channels, in_channels, kernel_size=3, stride=2, padding=1)
        )


    def forward(self, x, *args):
        return self.downsample(x)

class UpSample(nn.Module):
    def __init__(self, in_channels):
        super().__init__()

        self.upsample = nn.Sequential(
            nn.Upsample(scale_factor=2, mode="nearest"),
            nn.Conv2d(in_channels=in_channels, out_channels=in_channels, kernel_size=3, stride=1, padding=1),
        )

    def forward(self, x, *args):
        return self.upsample(x)

class Unet(nn.Module):
    def __init__(self, input_channels=3, output_channels=3, num_res_blocks=2,
                base_channels=128, base_channels_mult=(1, 2, 4, 8),
                apply_attention= (False, True, True, False),
                dropout_rate = 0.1,
                time_multiple=4):

        super().__init__()

        time_emb_dim_exp = base_channels * time_multiple
        self.time_embeddings = SinusoidalPositionEmbeddings(
            timesteps = 1000,
            embedding_dim = base_channels,
            embedding_dim_exp = time_emb_dim_exp
        )

        self.first = nn.Conv2d(input_channels, base_channels, kernel_size=3, stride=1, padding="same")
        num_resolutions = len(base_channels_mult)

        # Encoder
        self.encoder_blocks = nn.ModuleList()
        current_channels = [base_channels]
        in_channels = base_channels

        for i in range(num_resolutions):
            out_channels = base_channels * base_channels_mult[i]

            for _ in range(num_res_blocks):
                block = ResNetBlock(
                    in_channels=in_channels,
                    out_channels=out_channels,
                    dropout_rate=dropout_rate,
                    time_emb_dims=time_emb_dim_exp,
                    apply_attention=apply_attention[i]
                )
                self.encoder_blocks.append(block)
                in_channels = out_channels
                current_channels.append(in_channels)

            if i != (num_resolutions - 1):
                self.encoder_blocks.append(DownSample(in_channels))
                current_channels.append(in_channels)
                #current_channels.append(in_channels)

        # Bottleneck

        self.bottleneck_blocks = nn.ModuleList(
            (
                ResNetBlock(
                    in_channels=in_channels,
                    out_channels=in_channels,
                    dropout_rate=dropout_rate,
                    time_emb_dims=time_emb_dim_exp,
                    apply_attention=True,
                ),
                ResNetBlock(
                    in_channels=in_channels,
                    out_channels=in_channels,
                    dropout_rate=dropout_rate,
                    time_emb_dims=time_emb_dim_exp,
                    apply_attention=False,
                ),
            )
        )

        # Decoder
        self.decoder_blocks = nn.ModuleList()
        for i in reversed(range(num_resolutions)):
            out_channels = base_channels * base_channels_mult[i]

            for _ in range(num_res_blocks + 1):
                encoder_in_channels = current_channels.pop()
                block = ResNetBlock(
                    in_channels=encoder_in_channels + in_channels,#+ in_channels,
                    out_channels=out_channels,
                    dropout_rate=dropout_rate,
                    time_emb_dims=time_emb_dim_exp,
                    apply_attention=apply_attention[i]
                )
                self.decoder_blocks.append(block)
                in_channels = out_channels

            if i != 0:
                self.decoder_blocks.append(UpSample(in_channels))
        
        self.final = nn.Sequential(
            nn.GroupNorm(8, in_channels),
            nn.SiLU(),
            nn.Conv2d(in_channels, 3, kernel_size=3, stride=1, padding="same"),
        )

    def forward(self, x, t):
        time_emb = self.time_embeddings(t)
        h = self.first(x)
        outs = [h]

        for layer in self.encoder_blocks:
            h = layer(h, time_emb)
            outs.append(h)

        outs_shape = [out.shape for out in outs]

        for layer in self.bottleneck_blocks:
            h = layer(h, time_emb)

        for layer in self.decoder_blocks:
            if isinstance(layer, ResNetBlock):
                out = outs.pop()
                h = torch.cat([h, out], dim=1)
            h = layer(h, time_emb)
        h = self.final(h)

        return h