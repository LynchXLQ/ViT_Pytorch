# -*- Coding: utf-8 -*-
# @Time     : 1/10/2023 6:34 PM
# @Author   : Linqi Xiao
# @Software : PyCharm
# @Version  : python 3.10
# @Description :

import torch
from torch import nn
from torchvision import transforms

from PIL import Image
from einops import rearrange, repeat, reduce
from einops.layers.torch import Rearrange, Reduce


# ==================================== Test ====================================

PATCH_SIZE = 16  # (P; P) is the resolution of each image patch
device = 'cuda' if torch.cuda.is_available() else 'cpu'
img = Image.open('./dog.jpg')
pipline = transforms.Compose([
    transforms.Resize(size=(224, 224)),
    transforms.ToTensor()
])

X = pipline(img)  # [3, 224, 224]
X = X.unsqueeze(axis=0)  # [1, 3, 224, 224], add batch dim
print(X.shape)

patches = rearrange(X, pattern='b c (h s1) (w s2) -> b (h w) (s1 s2 c)', s1=PATCH_SIZE, s2=PATCH_SIZE)  # [1, 196, 768]

# ==================================== Test ====================================


def pair(t):
    return t if isinstance(t, tuple) else (t, t)


class PatchEmbedding(nn.Module):
    def __init__(self, image_size=224, patch_size=16, in_channels=3, emb_size=768):
        super(PatchEmbedding, self).__init__()
        image_height, image_width = pair(image_size)   # 224, 224
        patch_height, patch_width = pair(patch_size)   # 16, 16
        num_patches = (image_height // patch_height) * (image_width // patch_width)   # 196
        patch_dim = in_channels * patch_height * patch_width   # 3 * 16 * 16 = 768
        self.projection = nn.Sequential(
            Rearrange(pattern='b c (h s1) (w s2) -> b (h w) (s1 s2 c)', s1=patch_height, s2=patch_width),
            nn.Linear(in_features=patch_dim, out_features=emb_size)
        )
        self.cls_token = nn.Parameter(torch.randn(1, 1, emb_size))
        self.pos_embedding = nn.Parameter(torch.randn(1, num_patches + 1, emb_size))

    def forward(self, x):
        x = self.projection(x)
        b, n, _ = x.shape  # [b, 196, 768]
        # adding an extra learnable “classification token” to the sequence
        cls_token = repeat(self.cls_token, pattern='1 1 d -> b 1 d', b=b)
        x = torch.cat([cls_token, x], dim=1)  # [b, 196, 768] -> [b, 197, 768]
        x += self.pos_embedding
        return x


class MultiHeadAttention(nn.Module):
    # Scaled Dot-Product Attention
    def __init__(self, emb_size=768, heads=12, dim_head=64, dropout=0.):
        super(MultiHeadAttention, self).__init__()
        inner_dim = dim_head * heads  # 12 * 64 = 768
        self.heads = heads
        self.scale = dim_head ** -0.5  # 1/sqrt(dk)
        self.to_qkv = nn.Linear(emb_size, inner_dim * 3, bias=False)
        self.att_softmax = nn.Softmax(dim=-1)
        self.att_dropout = nn.Dropout(p=dropout)
        self.to_out = nn.Sequential(
            nn.Linear(in_features=inner_dim, out_features=emb_size),
            nn.Dropout(p=dropout)
        )

    def forward(self, x):
        qkv = self.to_qkv(x)
        qkv = rearrange(qkv, pattern='b n (h d qkv) -> qkv b h n d', h=self.heads, qkv=3)
        queries, keys, values = qkv[0], qkv[1], qkv[2]  # [BATCH, HEADS, SEQUENCE_LEN, EMBEDDING_SIZE]  [1, 8, 197, 64]
        dots = torch.matmul(queries,
                            keys.transpose(dim0=-1, dim1=-2))  # [1, 8, 197, 64] * [1, 8, 64, 197] = [1, 8, 197, 197]
        scales = dots * self.scale
        softmax = self.att_softmax(scales)
        dropout = self.att_dropout(softmax)
        out = torch.matmul(dropout, values)  # [1, 8, 197, 197] * [1, 8, 197, 64] = [1, 8, 197, 64]
        out = rearrange(out, pattern='b h n d -> b n (h d)')  # [1, 8, 197, 64] -> [1, 197, 512]
        out = self.to_out(out)  # [1, 197, 512] -> [1, 197, 768]
        return out


class FeedForward(nn.Module):
    def __init__(self, emb_size=768, hidden_dim=2048, dropout=0.):
        super(FeedForward, self).__init__()
        self.feedforward = nn.Sequential(
            nn.Linear(emb_size, hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, emb_size),
            nn.Dropout(dropout)
        )

    def forward(self, x):
        x = self.feedforward(x)
        return x


class ResidualAdd(nn.Module):
    def __init__(self, fn):
        super(ResidualAdd, self).__init__()
        self.fn = fn

    def forward(self, x, **kwargs):
        res = x
        x = self.fn(x, **kwargs)
        x += res
        return x


class TransformerEncoderBlock(nn.Module):
    def __init__(self, emb_size=768, dropout=0., **kwargs):
        super(TransformerEncoderBlock, self).__init__()
        self.att_with_res = ResidualAdd(
            nn.Sequential(
                nn.LayerNorm(normalized_shape=emb_size),
                MultiHeadAttention(emb_size=emb_size, dropout=dropout, **kwargs),
                nn.Dropout(p=dropout)
            )
        )
        self.ff_with_res = ResidualAdd(
            nn.Sequential(
                nn.LayerNorm(normalized_shape=emb_size),
                FeedForward(emb_size=emb_size, dropout=dropout, **kwargs),
                nn.Dropout(p=dropout)
            )
        )

    def forward(self, x):
        x = self.att_with_res(x)
        x = self.ff_with_res(x)
        return x


class TransformerEncoder(nn.Module):
    def __init__(self, depth=12, **kwargs):
        super(TransformerEncoder, self).__init__()
        self.trans = nn.ModuleList([])
        for _ in range(depth):   # how many blocks of TransformerEncoderBlock
            self.trans.append(TransformerEncoderBlock(**kwargs))

    def forward(self, x):
        for tran in self.trans:
            x = tran(x)
        return x


class ClassificationHead(nn.Module):
    def __init__(self, emb_size: int = 768, n_classes: int = 1000):
        super(ClassificationHead, self).__init__()
        self.classifier = nn.Sequential(
            Reduce('b n e -> b e', reduction='mean'),
            nn.LayerNorm(emb_size),
            nn.Linear(emb_size, n_classes)
        )

    def forward(self, x):
        x = self.classifier(x)
        return x



class ViT(nn.Module):
    def __init__(self, in_channels: int = 3, patch_size: int = 16, emb_size: int = 768, img_size: int = 224, depth: int = 12, n_classes: int = 1000, **kwargs):
        super(ViT, self).__init__()
        self.patch_embedding = PatchEmbedding(image_size=img_size, patch_size=patch_size, in_channels=in_channels, emb_size=emb_size)
        self.transformer_encoder = TransformerEncoder(depth=depth, **kwargs)
        self.classification_head = ClassificationHead(emb_size=emb_size, n_classes=n_classes)

    def forward(self, x):
        x = self.patch_embedding(x)   # [1, 3, 224, 224] -> [1, 197, 768]
        x = self.transformer_encoder(x)   # [1, 197, 768] -> [1, 197, 768]
        x = self.classification_head(x)   # [1, 197, 768] -> [1, 1000]
        return x




if __name__ == '__main__':
    # p = PatchEmbedding()()   # [1, 197, 768]
    # m = MultiHeadAttention()(p)
    # print(m.shape)
    t = ViT()(X)


