import torch
import torch.nn as nn
import math

from text_encoder import CLIPTextEncoder

class SinusoidalTimeEmbeddings(nn.Module):
    def __init__(self, num_steps, time_emb_dim, hidden_dim):
        super(SinusoidalTimeEmbeddings, self).__init__()

        self.num_steps = num_steps
        self.time_emb_dim = time_emb_dim
        self.hidden_dim = hidden_dim

        half_dim = time_emb_dim // 2
        emb = math.log(10000) / (half_dim - 1)
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


class SelfAttentionBlock(nn.Module):
    def __init__(self, in_channels):
        super(SelfAttentionBlock, self).__init__()

        self.in_channels = in_channels
        self.group_norm = nn.GroupNorm(32, in_channels, eps=1e-6)
        self.multihead_attention = nn.MultiheadAttention(in_channels, num_heads=4, batch_first=True)

    def forward(self, x):
        B, _, H, W = x.shape
        h = self.group_norm(x)
        h = h.reshape(B, self.in_channels, H*W).swapaxes(1, 2)
        h, _ = self.multihead_attention(h, h, h)
        h = h.swapaxes(2, 1).reshape(B, self.in_channels, H, W)
        return x + h

class CrossAtentionBlock(nn.Module):
    def __init__(self, in_channels, context_dim ):
        super(CrossAtentionBlock, self).__init__()

        self.in_channels = in_channels
        self.context_dim = context_dim
        self.group_norm = nn.GroupNorm(32, in_channels, eps=1e-6)

        self.q_proj = nn.Linear(in_channels, in_channels)
        self.k_proj = nn.Linear(context_dim, in_channels)
        self.v_proj = nn.Linear(context_dim, in_channels)
        self.out_proj = nn.Linear(in_channels, in_channels)

        self.attention = nn.MultiheadAttention(in_channels, num_heads=4, batch_first=True)

    def forward(self, x, context):
        B, C, H, W = x.shape
        h = self.group_norm(x)
        q = h.view(B, C, H * W).transpose(1, 2)

        q = self.q_proj(q)
        k = self.k_proj(context)
        v = self.v_proj(context)

        qkv_attn, _ = self.attention(q, k, v)
        out = self.out_proj(qkv_attn)

        out = out.transpose(1, 2).view(B, C, H, W)

        return x + out


class DownSampleBlock(nn.Module):
    def __init__(self, in_channels):
        super(DownSampleBlock, self).__init__()
        self.conv = nn.Conv2d(in_channels, in_channels, kernel_size=3, stride=2, padding=1)

    def forward(self, x, *args):
        return self.conv(x)
    
class UpSampleBlock(nn.Module):
    def __init__(self, in_channels):
        super(UpSampleBlock, self).__init__()

        self.upsample = nn.Sequential(
            nn.Upsample(scale_factor=2, mode="nearest"),
            nn.Conv2d(in_channels, in_channels, kernel_size=3, stride=1, padding=1),
        )

    def forward(self, x, *args):
        return self.upsample(x)
    

class ResidualBlock(nn.Module):
    def __init__(self, in_channels, out_channels, time_emb_dim, dropout_rate, apply_attention=False):
        super(ResidualBlock, self).__init__()

        self.in_channels = in_channels
        self.out_channels = out_channels
        self.time_emb_dim = time_emb_dim
        self.apply_attention = apply_attention
        self.dropout_rate = dropout_rate

        self.group_norm1 = nn.GroupNorm(32, in_channels, eps=1e-5)
        self.act = nn.SiLU()
        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=1, padding=1)

        self.group_norm2 = nn.GroupNorm(32, out_channels, eps=1e-5)
        self.conv2 = nn.Conv2d(out_channels, out_channels, kernel_size=3, stride=1, padding=1)
        self.dropout = nn.Dropout(dropout_rate)

        self.time_emb_match = nn.Linear(time_emb_dim, out_channels)

        if self.in_channels != self.out_channels:
            self.skip_connection = nn.Conv2d(in_channels, out_channels, kernel_size=1, stride=1)
        else:
            self.skip_connection = nn.Identity()

        if apply_attention:
            self.attention = CrossAtentionBlock(out_channels, 512)
        else:
            self.attention = nn.Identity()

    def forward(self, x, t, context):
        h = x

        h = self.group_norm1(h)
        h = self.act(h)
        h = self.conv1(h)

        t = self.time_emb_match(self.act(t))
        h += t[:, :, None, None]

        h = self.group_norm2(h)
        h = self.act(h)
        h = self.dropout(h)
        h = self.conv2(h)

        h = h + self.skip_connection(x)

        if isinstance(self.attention, CrossAtentionBlock):
            h = self.attention(h, context)
        return h

class UNet(nn.Module):
    def __init__(self, in_channels, out_channels, hidden_dim, time_emb_dim, time_emb_hidden_dim, context_dim, num_blocks, num_resnet_blocks,
                 in_channels_mult, decoder_channels_mult, dropout_rate,
                 device):
        super(UNet, self).__init__()

        self.in_channels = in_channels
        self.out_channels = out_channels
        self.hidden_dim = hidden_dim
        self.time_emb_dim = time_emb_dim
        self.context_dim = context_dim
        self.num_blocks = num_blocks
        self.in_channels_mult = in_channels_mult
        self.decoder_channels_mult = decoder_channels_mult
        self.num_resnet_blocks = num_resnet_blocks
        self.apply_attention_block = (False, True, True, False)

        self.text_encoder = CLIPTextEncoder(device=device)
        self.time_encoder = SinusoidalTimeEmbeddings(num_steps=1000, time_emb_dim=time_emb_dim, hidden_dim=time_emb_hidden_dim)

        self.first_layer = nn.Conv2d(in_channels, hidden_dim, kernel_size=3, stride=1, padding="same")

        self.downsample_blocks = nn.ModuleList()
        

        self.original_in_channels = hidden_dim
        in_channels = hidden_dim
        self.encoder_channels = [in_channels]

        for num_block in range(self.num_blocks):
            hidden_dim *= self.in_channels_mult[num_block]

            for _ in range(self.num_resnet_blocks):
                resnet_block = ResidualBlock(in_channels, hidden_dim, time_emb_hidden_dim, dropout_rate, apply_attention=self.apply_attention_block[num_block])
                self.downsample_blocks.append(resnet_block)
                in_channels = hidden_dim
                self.encoder_channels.append(in_channels)

            if num_block != self.num_blocks - 1:
                self.downsample_blocks.append(DownSampleBlock(in_channels))
                self.encoder_channels.append(in_channels)

        self.bottleneck_blocks = nn.ModuleList()
        self.bottleneck_blocks.append(ResidualBlock(in_channels, in_channels, time_emb_hidden_dim, dropout_rate, apply_attention=True))
        self.bottleneck_blocks.append(ResidualBlock(in_channels, in_channels, time_emb_hidden_dim, dropout_rate, apply_attention=False))

        in_channels = hidden_dim

        self.upsample_blocks = nn.ModuleList()

        for num_block in reversed(range(self.num_blocks)):
            hidden_dim = self.original_in_channels * self.decoder_channels_mult[num_block]

            for _ in range(self.num_resnet_blocks + 1):
                encoder_match_channels = self.encoder_channels.pop()
                resnet_block = ResidualBlock(in_channels + encoder_match_channels, hidden_dim, time_emb_hidden_dim, dropout_rate, self.apply_attention_block[num_block])
                self.upsample_blocks.append(resnet_block)
                in_channels = hidden_dim

            if num_block != 0:
                self.upsample_blocks.append(UpSampleBlock(in_channels))

        self.last_layer = nn.Sequential(nn.GroupNorm(8, in_channels),
                                        nn.SiLU(), nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=1, padding="same"))


    def forward(self, x, t, context_embeddings):
        time_embeddings = self.time_encoder(t)
        #context_embeddings = self.text_encoder(context)

        h = x
        h = self.first_layer(h)

        downsample_outs = [h]

        for layer in self.downsample_blocks:
            h = layer(h, time_embeddings, context_embeddings)
            downsample_outs.append(h)

        for layer in self.bottleneck_blocks:
            h = layer(h, time_embeddings, context_embeddings)

        for i , layer in enumerate(self.upsample_blocks):
            if isinstance(layer, ResidualBlock):
                downsample_out = downsample_outs.pop()
                h = torch.cat((h, downsample_out), dim=1)
            h = layer(h, time_embeddings, context_embeddings)

        h = self.last_layer(h)
        return h
    

# fake_image = torch.randn(1, 4, 32, 32)
# prompt = "A beautiful landscape with mountains and a river"
# prompt = CLIPTextEncoder()(prompt)
# t = torch.randint(0, 1000, (1, ), device="cuda")

# unet = UNet(in_channels=4, out_channels=3, hidden_dim=320, time_emb_dim=64, time_emb_hidden_dim=512, context_dim=77, num_blocks=4, num_resnet_blocks=2, in_channels_mult=(1,2,2,1),
#             decoder_channels_mult=(1,2,4,4,), dropout_rate=0.1, device="cuda")
# unet = unet.to("cuda")
# fake_image = fake_image.to("cuda")


# h = unet(fake_image, t, prompt)



