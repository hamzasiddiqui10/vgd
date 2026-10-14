import torch
import torch.nn as nn
import torch.nn.functional as F
import math

class FusedProbe(nn.Module):
    """Takes a trajectory model and a vision model, freezes them, discards the final linear layer
    and trains a new one on top (similar to "Chirality in Action")
    """

    def __init__(self, traj_encoder, vis_encoder, num_classes):
        super().__init__()

        self.traj_encoder = traj_encoder
        self.vis_encoder = vis_encoder
        self.dim_embed = vis_encoder.dim_embed + traj_encoder.dim_embed
        self.num_classes = num_classes

        # freeze both backbones
        for param in self.traj_encoder.parameters():
            param.requires_grad = False
        for param in self.vis_encoder.parameters():
            param.requires_grad = False

        self.linear = nn.Linear(self.dim_embed, self.num_classes)


    def forward(self, data):
        with torch.no_grad():
            _, traj_feat = self.traj_encoder.forward_feat(data)
            vis_feat = self.vis_encoder.forward_feat(data)
        
        # concatenate [B, dim_traj + dim_dino]; norm is needed for training dynamics
        traj_feat = F.layer_norm(traj_feat, [traj_feat.shape[-1]])
        vis_feat = F.layer_norm(vis_feat, [vis_feat.shape[-1]])

        fused = torch.cat([traj_feat, vis_feat], dim=-1)

        pred = self.linear(fused)
        return pred


class AttentiveProbe(nn.Module):
    """Attentive probe similar to V-JEPA, but with an additional absolute positional encoding for image models
    """
    def __init__(self, dim_embed, num_frames=16, spatial_size=16, num_heads=6, pos_enc=True, feature_key='dino_feat', num_classes=174):
        super().__init__()
        self.dim_embed = dim_embed
        self.feature_key = feature_key
        self.num_classes = num_classes
        self.query = nn.Parameter(torch.zeros(1, 1, dim_embed))
        nn.init.trunc_normal_(self.query, std=0.02)

        self.norm_attn = nn.LayerNorm(dim_embed)
        self.attn = nn.MultiheadAttention(
            embed_dim=dim_embed,
            num_heads=num_heads,
            batch_first=True,
        )

        self.norm_mlp = nn.LayerNorm(dim_embed)
        self.mlp = nn.Sequential(
            nn.Linear(dim_embed, dim_embed * 4),
            nn.GELU(),
            nn.Linear(dim_embed * 4, dim_embed),
        )

        self.linear = nn.Linear(self.dim_embed, self.num_classes)

        if pos_enc:
            pos = _get_3d_sincos_pos_embed(num_frames, spatial_size,
                                        spatial_size, dim_embed)
            self.register_buffer('pos_embed', pos.unsqueeze(0))  # [1, T*H*W, D]
        else:
            self.pos_embed = 0
    
    def forward_feat(self, data):
        assert self.feature_key in data, f'"{self.feature_key}" does not exist in the data.'

        x = data[self.feature_key]              # [B, T, D] / [B, T, HW, D]
        if x.dim() == 4:
            # x = x[:, torch.randperm(x.shape[1])]  # random temporal order
            # x = x[:,8].unsqueeze(1)               # center frame
            x = x.flatten(1,2)
        B = x.shape[0]                          # [B, T, D]
        x = x + self.pos_embed

        query = self.query.expand(B, -1, -1)    # [B, 1, D]

        # Cross-attention block (pre-norm)
        x = self.norm_attn(x)
        query = query + self.attn(query, x, x)[0]

        # MLP block (pre-norm)
        query = query + self.mlp(self.norm_mlp(query))

        return query.squeeze(1)                  # [B, D]

    def forward(self, data):
        feat = self.forward_feat(data)
        return self.linear(feat)
    

class AttentiveProbeTraj(AttentiveProbe):
    """Attentive probe for trajectory model
    """
    def __init__(self, dim_embed, num_frames=16, spatial_size=16, num_heads=6, pos_enc=True, feature_key=None, traj_enc=None, num_classes=0):
        super().__init__(dim_embed, num_frames, spatial_size, num_heads, pos_enc, feature_key, num_classes)
        self.traj_encoder = traj_enc

        for param in self.traj_encoder.parameters():
            param.requires_grad = False
    
    def forward_feat(self, data):
        with torch.no_grad():
            x, _ = self.traj_encoder.forward_feat(data)
        B = x.shape[0]                          # [B, T, D]
        x = x + self.pos_embed

        query = self.query.expand(B, -1, -1)    # [B, 1, D]

        # Cross-attention block (pre-norm)
        x = self.norm_attn(x)
        query = query + self.attn(query, x, x)[0]

        # MLP block (pre-norm)
        query = query + self.mlp(self.norm_mlp(query))

        return query.squeeze(1)                  # [B, D]

    def forward(self, data):
        feat = self.forward_feat(data)
        return self.linear(feat)
    

class LinearProbe(nn.Module):
    def __init__(self, model=None, dim_embed=288, num_classes=2, feature_key='dino_feat'):
        super().__init__()
        self.feature_key = feature_key
        self.linear = nn.Linear(dim_embed*2500, num_classes)
        self.num_classes = num_classes
        self.model = model
        if self.model is not None:
            for param in self.model.parameters():
                param.requires_grad = False

    def forward_feat(self, data):
        with torch.no_grad():
            if self.model is not None:
                self.model.eval()
                x, _ = self.model.forward_feat(data)
                if x.dim() > 2:
                    x = x.flatten(1)
                    pass
            else:
                x = data[self.feature_key]
                x = x.mean(dim=2).squeeze(1)
        return x

    def forward(self, data):
        x = self.forward_feat(data)
        return self.linear(x)
    

class NoProbe(nn.Module):
    def __init__(self, feature_key='dino_feat'):
        super().__init__()
        self.feature_key = feature_key

    def forward(self, data):
        return data[self.feature_key]
    
    def forward_feat(self, data):
        return data[self.feature_key], data[self.feature_key]


def _get_1d_sincos_pos_embed(length, dim):
    """Standard 1-D sinusoidal embedding.  Returns (length, dim)."""
    pos = torch.arange(length, dtype=torch.float32).unsqueeze(1)
    div = torch.exp(
        torch.arange(0, dim, 2, dtype=torch.float32)
        * (-math.log(10000.0) / dim)
    )
    pe = torch.zeros(length, dim)
    pe[:, 0::2] = torch.sin(pos * div)
    pe[:, 1::2] = torch.cos(pos * div)
    return pe


def _get_3d_sincos_pos_embed(T, H, W, dim):
    """Additive 3-D sinusoidal embedding.  Returns (T*H*W, dim)."""
    pe_t = _get_1d_sincos_pos_embed(T, dim)
    pe_h = _get_1d_sincos_pos_embed(H, dim)
    pe_w = _get_1d_sincos_pos_embed(W, dim)
    pe = (pe_t[:, None, None, :]
          + pe_h[None, :, None, :]
          + pe_w[None, None, :, :])
    return pe.reshape(T * H * W, dim)
