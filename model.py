import torch
from torch import nn
import numpy as np
import torch.nn.functional as F
import math


class AttnPooling(nn.Module):

    def __init__(self, emb_dim):
       super().__init__()
       self.dense1 = nn.Linear(emb_dim, emb_dim)
       self.dense2 = nn.Linear(emb_dim, emb_dim)
       self.relu = nn.ReLU()
       self.softmax = nn.Softmax(1)
    
    def forward(self, x, mask):
       bsz, ch, emb = x.size()
       residual = x

       # Dense 1
       x = x.reshape(bsz * ch, emb)
       x = self.relu(self.dense1(x))
       x = x.reshape(bsz, ch, emb)

       # Dense 2
       x = x.reshape(bsz * ch, emb)
       x = self.dense2(x)
       x = x.reshape(bsz,  ch, emb)

       x *= mask
       x = self.softmax(x)
       x = torch.sum(x * residual, 1)
       return x


class MaskedEncoderBlock(nn.Module):
    def __init__(self, n_emb, n_heads, dropout=.01, mask_type=None):
       super().__init__()
       self.n_head = n_heads
       self.key = nn.Linear(n_emb, n_emb)
       self.query = nn.Linear(n_emb, n_emb)
       self.value = nn.Linear(n_emb, n_emb)

       self.dropout = nn.Dropout(dropout)
       self.layer_norm = nn.LayerNorm(n_emb, eps=1e-6)

       self.proj = nn.Linear(n_emb, n_emb)
       self.fc = nn.Linear(n_emb, n_emb)
       self.mask_type = mask_type

    def forward(self, x, mask):

        B, T, C = x.size()
        residual = x

        k = self.key(x).view(B, T, self.n_head, C // self.n_head).transpose(1, 2) # (B, nh, T, hs)
        q = self.query(x).view(B, T, self.n_head, C // self.n_head).transpose(1, 2) # (B, nh, T, hs)
        v = self.value(x).view(B, T, self.n_head, C // self.n_head).transpose(1, 2) # (B, nh, T, hs)

        att = (q @ k.transpose(-2, -1)) * (1.0 / math.sqrt(k.size(-1)))

        if self.mask_type == "diag":
            attn_mask = torch.ones([T, T]).triu().tril() * float("-inf")
            attn_mask = attn_mask.reshape(1, 1, T, T,)
            attn_mask[torch.isnan(attn_mask)] = 0
        elif self.mask_type == "forward":
            attn_mask = torch.ones([T, T]).tril(diagonal=-1) * float("-inf")
            attn_mask = attn_mask.reshape(1, 1, T, T, )
            attn_mask[torch.isnan(attn_mask)] = 0
        else:
            attn_mask = torch.ones([T, T]).triu(diagonal=1) * float("-inf")
            attn_mask = attn_mask.reshape(1, 1, T, T, )
            attn_mask[torch.isnan(attn_mask)] = 0
        att += attn_mask

        att = F.softmax(att, dim=-1)
        att = self.dropout(att)
        y = att @ v   # (B, nh, T, T) x (B, nh, T, hs) -> (B, nh, T, hs)
        y = y.transpose(1, 2).contiguous().view(B, T, C)
        y = self.proj(y)
        y *= mask

        x = self.layer_norm(y) + residual
        residual = x
        x = self.fc(x)
        x = self.layer_norm(x) + residual
        return x


class BiteNet(nn.Module):

    def __init__(self, embedding_dim, output_dim, n_heads, blocks, n_visits, n_codes):
      super().__init__()

      # embedding layers
      self.emb = nn.Embedding(n_visits * n_codes, embedding_dim)
      self.int_emb = nn.Embedding(n_visits, embedding_dim)

      self.masc_enc_diag = MaskedEncoderBlock(embedding_dim, n_heads, mask_type="diag")
      self.masc_enc_forward = MaskedEncoderBlock(embedding_dim, n_heads, mask_type="forward")
      self.masc_enc_backward = MaskedEncoderBlock(embedding_dim, n_heads, mask_type="backward")

      # weird attn layers
      self.attn_pool1 = AttnPooling(embedding_dim)
      self.attn_pool2 = AttnPooling(embedding_dim)
      self.attn_pool3 = AttnPooling(embedding_dim)

      # fully Connected Layers
      self.fc1 = nn.Linear(embedding_dim * 2, embedding_dim)
      self.logits = nn.Linear(embedding_dim, output_dim)
      self.relu = nn.ReLU()
      self.sig = nn.Sigmoid()

    def forward(self, x, intervals):
      bsz, visits, codes, = x.size()

      base_mask = torch.ne(x, 0).float()
      base_mask_v = torch.sum(base_mask, -1)

      input_mask = base_mask
      input_mask = input_mask.reshape(bsz * visits, codes, 1)

      input_mask_v = base_mask_v
      input_mask_v = input_mask_v.reshape(bsz, visits, 1)

      emb = self.emb(x).reshape(bsz * visits, codes, -1)
      int_emb = self.int_emb(intervals)

      x = self.masc_enc_diag(emb, input_mask)
      x = self.attn_pool1(x, input_mask)
      x = x.reshape(bsz, visits, -1)
      x = x + int_emb

      y = self.masc_enc_forward(x, input_mask_v)
      y = self.attn_pool2(y, input_mask_v)

      z = self.masc_enc_backward(x, input_mask_v)
      z = self.attn_pool3(z, input_mask_v)

      x = torch.cat([y, z], dim=-1)
      x = self.relu(self.fc1(x))
      x = self.sig(self.logits(x))
      return x