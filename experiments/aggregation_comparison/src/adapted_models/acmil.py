"""ACMIL-GA patient adaptation with bag-aware stochastic top-K masking.

Derived from dazhangyu123/ACMIL e53d19c, MIT, Copyright 2022 benbergner.
See docs/licenses/ACMIL-LICENSE.md. No upstream training code is executed.
"""
import math
import torch
from torch import nn
from torch.nn import functional as F
from .nystrom import compact_bag

class ACMILPatient(nn.Module):
    def __init__(self,input_dim=1280,hidden_dim=256,n_branches=5,n_masked_patch=10,mask_drop=.6):
        super().__init__()
        if not 0<=mask_drop<=1 or n_branches<2:raise ValueError('Invalid ACMIL settings')
        self.config=dict(input_dim=input_dim,hidden_dim=hidden_dim,n_branches=n_branches,
            n_masked_patch=n_masked_patch,mask_drop=mask_drop,branch_ce_weight=1.,
            bag_ce_weight=1.,attention_diversity_weight=1.,classification='two-class CE')
        self.projection=nn.Sequential(nn.Linear(input_dim,hidden_dim,bias=False),nn.ReLU())
        self.v=nn.Linear(hidden_dim,128);self.u=nn.Linear(hidden_dim,128)
        self.w=nn.Linear(128,n_branches)
        self.branch_heads=nn.ModuleList([nn.Linear(hidden_dim,2) for _ in range(n_branches)])
        self.bag_head=nn.Linear(hidden_dim,2)
        self.n_branches,self.n_masked_patch,self.mask_drop=n_branches,n_masked_patch,mask_drop

    def forward(self,x,valid_mask=None):
        x,valid=compact_bag(x,valid_mask)
        h=self.projection(x)
        scores=self.w(torch.tanh(self.v(h))*torch.sigmoid(self.u(h))).T
        keep=torch.ones_like(scores,dtype=torch.bool)
        candidates=min(self.n_masked_patch,max(0,len(x)-1))
        n_drop=min(len(x)-1,math.floor(candidates*self.mask_drop)) if self.training else 0
        if n_drop:
            top=scores.topk(candidates,dim=-1).indices
            random_order=torch.rand_like(top,dtype=scores.dtype).argsort(-1)[:,:n_drop]
            dropped=top.gather(1,random_order)
            keep.scatter_(1,dropped,False)
        attention=scores.masked_fill(~keep,-torch.finfo(scores.dtype).max).softmax(-1)
        features=attention@h
        branches=torch.stack([head(features[i]) for i,head in enumerate(self.branch_heads)])
        bag_feature=(attention.mean(0,keepdim=True)@h).squeeze(0)
        return {'logits':self.bag_head(bag_feature),'branch_logits':branches,'attention':attention,
            'keep_mask':keep,'candidate_count':candidates,'masked_count':n_drop,'input_valid_mask':valid}

    def objective(self,output,label,epoch=1,update_centers=False):
        y=torch.as_tensor(label,dtype=torch.long,device=output['logits'].device).reshape(1)
        bag=F.cross_entropy(output['logits'].reshape(1,2),y)
        branch=F.cross_entropy(output['branch_logits'],y.expand(self.n_branches))
        a=F.normalize(output['attention'],dim=-1)
        pairwise=a@a.T
        indices=torch.triu_indices(self.n_branches,self.n_branches,offset=1,device=a.device)
        diversity=pairwise[indices[0],indices[1]].mean()
        return bag+branch+diversity,{'classification':bag,'branch_ce':branch,'attention_diversity':diversity}
