"""Minimal SeHGNN2 knowledge model closure used by CHORD."""

import math

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_sparse import SparseTensor


def xavier_uniform_(tensor, gain=1.):
    fan_in, fan_out = tensor.size()[-2:]
    std = gain * math.sqrt(2.0 / float(fan_in + fan_out))
    a = math.sqrt(3.0) * std
    return torch.nn.init._no_grad_uniform_(tensor, -a, a)


class Transformer(nn.Module):
    """Transformer-based semantic fusion in SeHGNN."""

    def __init__(self, n_channels, num_heads=1, att_drop=0., act='none'):
        super().__init__()
        self.n_channels = n_channels
        self.num_heads = num_heads
        assert self.n_channels % (self.num_heads * 4) == 0

        self.query = nn.Linear(self.n_channels, self.n_channels // 4)
        self.key = nn.Linear(self.n_channels, self.n_channels // 4)
        self.value = nn.Linear(self.n_channels, self.n_channels)

        self.gamma = nn.Parameter(torch.tensor([0.]))
        self.att_drop = nn.Dropout(att_drop)
        if act == 'sigmoid':
            self.act = torch.nn.Sigmoid()
        elif act == 'relu':
            self.act = torch.nn.ReLU()
        elif act == 'leaky_relu':
            self.act = torch.nn.LeakyReLU(0.2)
        elif act == 'none':
            self.act = lambda x: x
        else:
            assert 0, f'Unrecognized activation function {act} for class Transformer'

        self.reset_parameters()

    def reset_parameters(self):
        for k, v in self._modules.items():
            if hasattr(v, 'reset_parameters'):
                v.reset_parameters()
        nn.init.zeros_(self.gamma)

    def forward(self, x, mask=None):
        B, M, C = x.size()
        H = self.num_heads
        if mask is not None:
            assert mask.size() == torch.Size((B, M))

        f = self.query(x).view(B, M, H, -1).permute(0, 2, 1, 3)
        g = self.key(x).view(B, M, H, -1).permute(0, 2, 3, 1)
        h = self.value(x).view(B, M, H, -1).permute(0, 2, 1, 3)

        beta = F.softmax(self.act(f @ g / math.sqrt(f.size(-1))), dim=-1)
        beta = self.att_drop(beta)
        if mask is not None:
            beta = beta * mask.view(B, 1, 1, M)
            beta = beta / (beta.sum(-1, keepdim=True) + 1e-12)

        o = self.gamma * (beta @ h)
        return o.permute(0, 2, 1, 3).reshape((B, M, C)) + x


class LinearPerMetapath(nn.Module):
    """Linear projection per metapath for feature projection in SeHGNN."""

    def __init__(self, cin, cout, num_metapaths):
        super().__init__()
        self.cin = cin
        self.cout = cout
        self.num_metapaths = num_metapaths

        self.W = nn.Parameter(torch.randn(self.num_metapaths, self.cin, self.cout))
        self.bias = nn.Parameter(torch.zeros(self.num_metapaths, self.cout))

        self.reset_parameters()

    def reset_parameters(self):
        gain = nn.init.calculate_gain("relu")
        xavier_uniform_(self.W, gain=gain)
        nn.init.zeros_(self.bias)

    def forward(self, x):
        return torch.einsum('bcm,cmn->bcn', x, self.W) + self.bias.unsqueeze(0)


unfold_nested_list = lambda x: sum(x, [])


class TokenEncoder2(nn.Module):
    """Encode token-level features into node-level features."""

    def __init__(self, token_feat_dim, hidden_dim, num_heads=4, dropout=0.1):
        super().__init__()
        self.self_attn = nn.MultiheadAttention(
            embed_dim=token_feat_dim,
            num_heads=num_heads,
            dropout=dropout,
            batch_first=True
        )
        self.norm = nn.LayerNorm(token_feat_dim)
        self.projection = nn.Sequential(
            nn.Linear(token_feat_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.ReLU()
        )

    def forward(self, x, mask=None):
        attn_output, _ = self.self_attn(x, x, x, key_padding_mask=mask)
        x = self.norm(attn_output + x)
        x_pooled = x.mean(dim=1)
        return self.projection(x_pooled)


class SeHGNN2(nn.Module):
    """The SeHGNN knowledge model (trimmed to the used feature path)."""

    def __init__(self, dataset, nfeat, hidden, nclass, feat_keys,
                 label_feat_keys, tgt_type, dropout, input_drop, att_drop,
                 n_fp_layers, n_task_layers, act, residual=False,
                 data_size=None, drop_metapath=0., num_heads=1,
                 token_feat_dim=768, token_encoder_heads=4, token_dropout=0.1):
        super().__init__()
        self.dataset = dataset
        self.feat_keys = sorted(feat_keys)
        self.label_feat_keys = sorted(label_feat_keys)
        self.num_channels = num_channels = (
            len(self.feat_keys) + len(self.label_feat_keys))
        self.tgt_type = tgt_type
        self.residual = residual

        self.token_encoders = nn.ModuleDict({})
        for k in feat_keys:
            self.token_encoders[k] = TokenEncoder2(
                token_feat_dim=token_feat_dim,
                hidden_dim=nfeat,
                num_heads=token_encoder_heads,
                dropout=token_dropout
            )

        self.input_drop = nn.Dropout(input_drop)

        self.data_size = data_size
        self.embeding = nn.ParameterDict({})
        for k, v in data_size.items():
            self.embeding[str(k)] = nn.Parameter(torch.Tensor(v, nfeat))

        if len(self.label_feat_keys):
            self.labels_embeding = nn.ParameterDict({})
            for k in self.label_feat_keys:
                self.labels_embeding[k] = nn.Parameter(torch.Tensor(nclass, nfeat))
        else:
            self.labels_embeding = {}

        self.feature_projection = nn.Sequential(
            *([LinearPerMetapath(nfeat, hidden, num_channels),
               nn.LayerNorm([num_channels, hidden]),
               nn.PReLU(),
               nn.Dropout(dropout), ]
              + unfold_nested_list([[
                        LinearPerMetapath(hidden, hidden, num_channels),
                        nn.LayerNorm([num_channels, hidden]),
                        nn.PReLU(),
                        nn.Dropout(dropout), ] for _ in range(n_fp_layers - 1)])
              )
        )

        self.semantic_fusion = Transformer(hidden, num_heads=num_heads,
                                           att_drop=att_drop, act=act)
        self.fc_after_concat = nn.Linear(num_channels * hidden, hidden)

        if self.residual:
            self.res_fc = nn.Linear(nfeat, hidden)

        self.task_mlp = nn.Sequential(
            *([nn.PReLU(),
               nn.Dropout(dropout), ]
              + unfold_nested_list([[
                        nn.Linear(hidden, hidden),
                        nn.BatchNorm1d(hidden, affine=False),
                        nn.PReLU(),
                        nn.Dropout(dropout), ] for _ in range(n_task_layers - 1)])
              + [nn.Linear(hidden, nclass),
                 nn.BatchNorm1d(nclass, affine=False,
                                track_running_stats=False)]
              )
        )

        self.reset_parameters()

    def reset_parameters(self):
        for k, v in self._modules.items():
            if isinstance(v, nn.ParameterDict):
                for _k, _v in v.items():
                    _v.data.uniform_(-0.5, 0.5)
            elif isinstance(v, nn.ModuleList):
                for block in v:
                    if isinstance(block, nn.Sequential):
                        for layer in block:
                            if hasattr(layer, 'reset_parameters'):
                                layer.reset_parameters()
                    elif hasattr(block, 'reset_parameters'):
                        block.reset_parameters()
            elif isinstance(v, nn.Sequential):
                for layer in v:
                    if hasattr(layer, 'reset_parameters'):
                        layer.reset_parameters()
            elif hasattr(v, 'reset_parameters'):
                v.reset_parameters()

    def forward_feat(self, feature_dict, label_dict={}, mask=None,
                     token_masks=None):
        processed_features = {}
        for k, x in feature_dict.items():
            if len(x.shape) == 3:
                token_mask = token_masks.get(k, None) if token_masks else None
                node_features = self.token_encoders[k](x, mask=token_mask)
                processed_features[k] = self.input_drop(node_features)
            else:
                if isinstance(x, torch.Tensor):
                    processed_features[k] = self.input_drop(
                        x @ self.embeding[k])
                elif isinstance(x, SparseTensor):
                    processed_features[k] = self.input_drop(
                        x @ self.embeding[k[-1]])
                else:
                    assert 0

        features = processed_features
        B = features[self.tgt_type].shape[0]

        labels = {k: self.input_drop(x @ self.labels_embeding[k])
                  for k, x in label_dict.items()}

        x = [features[k] for k in self.feat_keys] + [labels[k]
                                                     for k in self.label_feat_keys]
        x = torch.stack(x, dim=1)
        x = self.feature_projection(x)

        x = self.semantic_fusion(x, mask=None).transpose(1, 2)

        x = self.fc_after_concat(x.reshape(B, -1))
        if self.residual:
            x = x + self.res_fc(features[self.tgt_type])

        return x
