"""3-D vision transformer used by CHORD (trimmed to the ViT class)."""

import torch
from torch import nn

from einops import rearrange
from einops.layers.torch import Rearrange
from position_encoding import PositionEmbeddingLearned3d


# helpers

def pair(t):
    return t if isinstance(t, tuple) else (t, t)


# classes

class PreNorm(nn.Module):
    def __init__(self, dim, fn):
        super().__init__()
        self.norm = nn.LayerNorm(dim)
        self.fn = fn

    def forward(self, x, **kwargs):
        return self.fn(self.norm(x), **kwargs)


class FeedForward(nn.Module):
    def __init__(self, dim, hidden_dim, dropout=0.):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(dim, hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, dim),
            nn.Dropout(dropout)
        )

    def forward(self, x):
        return self.net(x)


class Attention(nn.Module):
    def __init__(self, dim, heads=8, dim_head=64, dropout=0.):
        super().__init__()
        inner_dim = dim_head * heads
        project_out = not (heads == 1 and dim_head == dim)

        self.heads = heads
        self.scale = dim_head ** -0.5

        self.attend = nn.Softmax(dim=-1)
        self.dropout = nn.Dropout(dropout)

        self.to_qkv = nn.Linear(dim, inner_dim * 3, bias=False)

        self.to_out = nn.Sequential(
            nn.Linear(inner_dim, dim),
            nn.Dropout(dropout)
        ) if project_out else nn.Identity()

    def forward(self, x):
        qkv = self.to_qkv(x).chunk(3, dim=-1)
        q, k, v = map(lambda t: rearrange(t, 'b n (h d) -> b h n d',
                                          h=self.heads), qkv)

        dots = torch.matmul(q, k.transpose(-1, -2)) * self.scale

        attn = self.attend(dots)
        self.attn_weights = attn  # kept from the original implementation
        attn = self.dropout(attn)

        out = torch.matmul(attn, v)
        out = rearrange(out, 'b h n d -> b n (h d)')
        return self.to_out(out)


class Transformer(nn.Module):
    def __init__(self, dim, depth, heads, dim_head, mlp_dim, dropout=0.):
        super().__init__()
        self.layers = nn.ModuleList([])
        for _ in range(depth):
            self.layers.append(nn.ModuleList([
                PreNorm(dim, Attention(dim, heads=heads, dim_head=dim_head,
                                       dropout=dropout)),
                PreNorm(dim, FeedForward(dim, mlp_dim, dropout=dropout))
            ]))

    def forward(self, x):
        for attn, ff in self.layers:
            x = attn(x) + x
            x = ff(x) + x
        return x


class ViTPooler(nn.Module):
    def __init__(self, hidden_size):
        super().__init__()
        self.dense = nn.Linear(hidden_size, hidden_size)
        self.activation = nn.Tanh()

    def forward(self, hidden_states):
        first_token_tensor = hidden_states[:, 0]
        pooled_output = self.dense(first_token_tensor)
        pooled_output = self.activation(pooled_output)
        return pooled_output


class ViT(nn.Module):
    def __init__(self, *, image_size, image_patch_size, frames, frame_patch_size,
                 dim, depth, heads, mlp_dim, pool='cls', channels=3,
                 dim_head=64, dropout=0., emb_dropout=0.):
        super().__init__()
        image_height, image_width = pair(image_size)
        patch_height, patch_width = pair(image_patch_size)
        self.image_size = image_height
        self.dim = dim
        self.depth = depth
        assert image_height % patch_height == 0 and image_width % patch_width == 0, \
            'Image dimensions must be divisible by the patch size.'
        assert frames % frame_patch_size == 0, \
            'Frames must be divisible by frame patch size'

        self.patch_height = patch_height
        self.patch_width = patch_width
        self.frame_patch_size = frame_patch_size

        self.num_patches = ((image_height // patch_height)
                            * (image_width // patch_width)
                            * (frames // frame_patch_size))
        self.patch_dim = (channels * patch_height * patch_width
                          * frame_patch_size)

        self.cls_token = nn.Parameter(torch.randn(1, 1, self.dim))

        assert pool in {'cls', 'mean'}, \
            'pool type must be either cls (cls token) or mean (mean pooling)'

        self.to_patch_embedding = nn.Sequential(
            Rearrange('b c (h p1) (w p2) (f pf) -> b (h w f) (p1 p2 pf c)',
                      p1=patch_height, p2=patch_width,
                      pf=self.frame_patch_size),
            nn.LayerNorm(self.patch_dim),
            nn.Linear(self.patch_dim, self.dim),
            nn.LayerNorm(self.dim),
        )

        self.pos_embedding = PositionEmbeddingLearned3d(
            dim // 3, (image_height // patch_height),
            (image_width // patch_width), (frames // frame_patch_size))
        self.cls_pos_embedding = nn.Parameter(torch.randn(1, 1, dim))
        self.dropout = nn.Dropout(emb_dropout)

        self.transformer = Transformer(dim, depth, heads, dim_head, mlp_dim,
                                       dropout)

        self.layernorm = nn.LayerNorm(self.dim, eps=1e-12)
        self.pooler = ViTPooler(self.dim)

    def embeddings(self, input):
        B, C, H, W, D = input.shape
        x = self.to_patch_embedding(input)
        b, n, _ = x.shape
        cls_tokens = self.cls_token.expand(b, -1, -1)
        cls_pos_embedding = self.cls_pos_embedding.expand(b, -1, -1)
        x = torch.cat([cls_tokens, x], dim=1)
        pos = self.pos_embedding(B, H // self.patch_height,
                                 W // self.patch_width,
                                 D // self.frame_patch_size, x)
        pos_add_cls = torch.cat([cls_pos_embedding, pos], dim=1)
        x += pos_add_cls
        x = self.dropout(x)
        return x, pos_add_cls

    def forward(self, x):
        x = self.transformer(x)
        return x
