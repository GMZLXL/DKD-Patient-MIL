"""ACMIL-GA-adapted-v2-official-mask: candidate ratio correction only.

Derived from dazhangyu123/ACMIL e53d19c, MIT, Copyright 2022 benbergner.
See docs/licenses/ACMIL-LICENSE.md. Historical ACMILPatient is unchanged.
"""
import math
import torch
from .acmil import ACMILPatient
from .nystrom import compact_bag


class ACMILOfficialRatio(ACMILPatient):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        if self.n_masked_patch < 0:
            raise ValueError('n_masked_patch must be nonnegative')
        self.config.update(variant='ACMIL-GA-adapted-v2-official-mask',
                           candidate_cap='K_valid', final_drop_cap='K_valid-1')

    def forward(self, x, valid_mask=None):
        x, valid = compact_bag(x, valid_mask)
        h = self.projection(x)
        scores = self.w(torch.tanh(self.v(h)) * torch.sigmoid(self.u(h))).T
        keep = torch.ones_like(scores, dtype=torch.bool)
        candidates = min(self.n_masked_patch, len(x))
        n_drop = min(len(x)-1, math.floor(candidates*self.mask_drop)) if self.training else 0
        if n_drop:
            top = scores.topk(candidates, dim=-1).indices
            random_order = torch.rand_like(top, dtype=scores.dtype).argsort(-1)[:, :n_drop]
            keep.scatter_(1, top.gather(1, random_order), False)
        attention = scores.masked_fill(~keep, -torch.finfo(scores.dtype).max).softmax(-1)
        features = attention @ h
        branches = torch.stack([head(features[i]) for i, head in enumerate(self.branch_heads)])
        bag_feature = (attention.mean(0, keepdim=True) @ h).squeeze(0)
        return {'logits': self.bag_head(bag_feature), 'branch_logits': branches,
                'attention': attention, 'keep_mask': keep, 'candidate_count': candidates,
                'masked_count': n_drop, 'input_valid_mask': valid}
