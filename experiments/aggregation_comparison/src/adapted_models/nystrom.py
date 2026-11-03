"""Mask-safe Nyström attention with exact attention for short sets.

Adapted from DGR-MIL models/nystrom_attention.py, MIT, Copyright 2024 Wen.
See docs/licenses/DGR-MIL-LICENSE.md and docs/method_adaptations.md.
"""
import math
import torch
from torch import nn
from torch.nn import functional as F

def compact_bag(x,valid_mask=None):
    if x.ndim!=2 or len(x)==0:raise ValueError('Expected nonempty [K,D] bag')
    if valid_mask is None:valid_mask=torch.ones(len(x),dtype=torch.bool,device=x.device)
    if valid_mask.dtype!=torch.bool or valid_mask.shape!=(len(x),):raise ValueError('Expected bool [K] mask')
    if not valid_mask.any():raise ValueError('All-padding bags are invalid')
    return x[valid_mask],valid_mask

def masked_softmax(sim,mask):
    # Explicit renormalization makes all-masked rows zero, never uniform/NaN.
    p=sim.masked_fill(~mask,-torch.finfo(sim.dtype).max).softmax(-1)*mask
    return p/p.sum(-1,keepdim=True).clamp_min(torch.finfo(sim.dtype).eps)

def iterative_pinv(x,iters=6):
    scale=x.abs().sum(-1).amax(-1,keepdim=True)*x.abs().sum(-2).amax(-1,keepdim=True)
    z=x.transpose(-1,-2)/scale.unsqueeze(-1).clamp_min(1e-8)
    identity=torch.eye(x.shape[-1],device=x.device,dtype=x.dtype)
    for _ in range(iters):
        xz=x@z
        z=.25*z@(13*identity-xz@(15*identity-xz@(7*identity-xz)))
    return z

class MaskedNystromAttention(nn.Module):
    def __init__(self,dim,heads=8,landmarks=128,dropout=.1,residual_kind='identity'):
        super().__init__()
        if dim%heads:raise ValueError('dim must be divisible by heads')
        self.heads,self.landmarks,self.dh=heads,landmarks,dim//heads
        self.qkv=nn.Linear(dim,3*dim,bias=False)
        self.out=nn.Sequential(nn.Linear(dim,dim),nn.Dropout(dropout))
        self.residual_kind=residual_kind
        self.conv=nn.Conv2d(heads,heads,(33,1),padding=(16,0),groups=heads,bias=False) if residual_kind=='conv' else None
        self.last_diagnostics=[]

    def _single(self,x):
        n,d=x.shape
        q,k,v=self.qkv(x).reshape(n,3,self.heads,self.dh).permute(1,2,0,3).unbind(0)
        q=q*(self.dh**-.5)
        if n<=self.landmarks:
            # One landmark per valid instance: exact softmax avoids an oversized
            # singular landmark system and any artificial tokens for K=1..9.
            attention=(q@k.transpose(-1,-2)).softmax(-1)
            out=attention@v
            mode='exact_short_bag';padding=0
        else:
            m=self.landmarks; length=math.ceil(n/m);padding=m*length-n
            qp=F.pad(q,(0,0,0,padding));kp=F.pad(k,(0,0,0,padding));vp=F.pad(v,(0,0,0,padding))
            valid=F.pad(torch.ones(n,dtype=torch.bool,device=x.device),(0,padding))
            counts=valid.reshape(m,length).sum(-1)
            lm=counts>0
            ql=qp.reshape(self.heads,m,length,self.dh).sum(-2)/counts.clamp_min(1)[None,:,None]
            kl=kp.reshape(self.heads,m,length,self.dh).sum(-2)/counts.clamp_min(1)[None,:,None]
            a=masked_softmax(qp@kl.transpose(-1,-2),valid[None,:,None]&lm[None,None,:])
            b=masked_softmax(ql@kl.transpose(-1,-2),lm[None,:,None]&lm[None,None,:])
            c=masked_softmax(ql@kp.transpose(-1,-2),lm[None,:,None]&valid[None,None,:])
            inv=iterative_pinv(b)
            out=((a@inv)@(c@vp))[:,:n]
            mode='masked_nystrom'
        if self.residual_kind=='identity':out=out+v
        elif self.conv is not None:out=out+self.conv(v.unsqueeze(0)).squeeze(0)
        self.last_diagnostics.append({'mode':mode,'valid_tokens':n,'internal_padding':padding})
        return self.out(out.transpose(0,1).reshape(n,d))

    def forward(self,x,valid_mask=None):
        if x.ndim!=3:raise ValueError('Expected [B,K,D]')
        self.last_diagnostics=[];outputs=[]
        for i in range(len(x)):
            valid=None if valid_mask is None else valid_mask[i]
            bag,mask=compact_bag(x[i],valid)
            y=self._single(bag)
            full=x.new_zeros(x[i].shape).index_copy(0,mask.nonzero().flatten(),y)
            outputs.append(full)
        return torch.stack(outputs)
