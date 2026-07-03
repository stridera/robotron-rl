"""slot_policy.py — entity-relational feature extractor for the 945-dim slot obs.

The obs is [player(2)] + [41 slots × 23 features]. A flat MLP destroys the per-entity
structure; the FSM reasons per-entity (find the relevant threat, act relative to it). This
extractor encodes each slot with a shared MLP, runs self-attention over the slots (masked
by the valid flag) so entities can be compared jointly, then pools — permutation-aware and
relational, matching the FSM's logic far better than a flattened MLP.

Use via policy_kwargs={'features_extractor_class': SlotAttnExtractor,
                       'features_extractor_kwargs': {'features_dim': 256}, 'net_arch':[256]}.
"""
import torch as th
import torch.nn as nn
from stable_baselines3.common.torch_layers import BaseFeaturesExtractor

N_SLOTS = 41
SLOT_F = 23
VALID_IDX = 16 + 2 + 1 + 1   # within-slot offset of the 'valid' flag (=20)


class SlotAttnExtractor(BaseFeaturesExtractor):
    def __init__(self, observation_space, features_dim: int = 256, embed: int = 64, heads: int = 4, layers: int = 2):
        super().__init__(observation_space, features_dim)
        self.slot_enc = nn.Sequential(nn.Linear(SLOT_F, embed), nn.ReLU(), nn.Linear(embed, embed), nn.ReLU())
        self.player_enc = nn.Sequential(nn.Linear(2, embed), nn.ReLU())
        enc_layer = nn.TransformerEncoderLayer(d_model=embed, nhead=heads, dim_feedforward=embed * 2,
                                               dropout=0.0, batch_first=True)
        self.attn = nn.TransformerEncoder(enc_layer, num_layers=layers)
        # output: [player token after attn] + [masked mean] + [masked max]  -> features_dim
        self.out = nn.Sequential(nn.Linear(embed * 3, features_dim), nn.ReLU())

    def forward(self, obs: th.Tensor) -> th.Tensor:
        B = obs.shape[0]
        player = obs[:, :2]
        slots = obs[:, 2:].reshape(B, N_SLOTS, SLOT_F)
        valid = slots[:, :, VALID_IDX] > 0.5                      # (B, 41)
        se = self.slot_enc(slots)                                 # (B, 41, E)
        pe = self.player_enc(player).unsqueeze(1)                 # (B, 1, E) — player token
        tokens = th.cat([pe, se], dim=1)                          # (B, 42, E)
        # key_padding_mask: True = ignore. Player token always attended; invalid slots masked.
        pad = th.cat([th.zeros(B, 1, dtype=th.bool, device=obs.device), ~valid], dim=1)
        enc = self.attn(tokens, src_key_padding_mask=pad)         # (B, 42, E)
        player_out = enc[:, 0]                                    # (B, E)
        slot_enc = enc[:, 1:]                                     # (B, 41, E)
        m = valid.unsqueeze(-1).float()
        mean = (slot_enc * m).sum(1) / m.sum(1).clamp(min=1.0)    # masked mean
        neg = th.where(valid.unsqueeze(-1), slot_enc, th.full_like(slot_enc, -1e9))
        mx = neg.max(1).values                                    # masked max
        return self.out(th.cat([player_out, mean, mx], dim=1))
