"""DGR-MIL-adapted: diverse global tokens, interaction, centers and objectives.

Derived from ChongQingNoSubway/DGR-MIL ac6257f, MIT, Copyright 2024 Wen.
See docs/licenses/DGR-MIL-LICENSE.md. No upstream training code is executed.
"""
import torch
from torch import nn
from torch.nn import functional as F
from .nystrom import MaskedNystromAttention,compact_bag

class Interaction(nn.Module):
    def __init__(self,d,ordered=False):
        super().__init__()
        self.norm=nn.LayerNorm(d)
        self.attention=MaskedNystromAttention(d,landmarks=d//2,residual_kind='conv' if ordered else 'identity')
        self.final_norm=nn.LayerNorm(d)
    def forward(self,x):return self.final_norm(x+self.attention(self.norm(x)))

class DGRMILPatient(nn.Module):
    def __init__(self,input_dim=1280,hidden_dim=256,n_global=5):
        super().__init__()
        self.config=dict(input_dim=input_dim,hidden_dim=hidden_dim,n_global=n_global,
            landmarks=hidden_dim//2,exact_short_bag=True,center_momentum=.4,
            auxiliary_start_epoch=10,diversity_weight=.1,alignment_weight=.1,
            gram_jitter=1e-6,classification='two-class CE')
        self.projection=nn.Sequential(nn.Linear(input_dim,hidden_dim),nn.ReLU(),nn.Linear(hidden_dim,hidden_dim),nn.ReLU())
        self.global_codes=nn.Parameter(torch.randn(1,n_global,input_dim))
        self.cls_token=nn.Parameter(torch.randn(1,1,hidden_dim))
        self.instances=Interaction(hidden_dim)
        self.globals=Interaction(hidden_dim,ordered=True)
        self.cross=nn.MultiheadAttention(hidden_dim,8,dropout=.1,batch_first=True)
        self.cross_ffn=nn.Sequential(nn.Linear(hidden_dim,hidden_dim),nn.LayerNorm(hidden_dim))
        self.classifier=nn.Linear(hidden_dim,2)
        self.register_buffer('positive_center',torch.randn(1,hidden_dim))
        self.register_buffer('negative_center',torch.randn(1,hidden_dim))
        self.register_buffer('center_updates',torch.zeros(2,dtype=torch.long))

    def forward(self,x,valid_mask=None):
        x,_=compact_bag(x,valid_mask)
        projected=self.projection(x)
        h=self.instances(projected.unsqueeze(0))
        diverse=self.projection(self.global_codes)
        global_tokens=self.globals(torch.cat((self.cls_token,diverse),dim=1))
        cross,attention=self.cross(global_tokens,h,h)
        feature=self.cross_ffn(cross)[:,0]
        return {'logits':self.classifier(feature).squeeze(0),'attention':attention,
                'center_feature':projected.mean(0,keepdim=True),'global_features':diverse.squeeze(0)}

    def objective(self,output,label,epoch=1,update_centers=True):
        if not self.training:raise RuntimeError('Training objective/center updates require train mode')
        y=torch.as_tensor(label,device=output['logits'].device,dtype=torch.long).reshape(1)
        ce=F.cross_entropy(output['logits'].reshape(1,2),y)
        if update_centers:
            with torch.no_grad():
                center=self.positive_center if int(y)==1 else self.negative_center
                center.mul_(.4).add_(output['center_feature'].detach(),alpha=.6)
                self.center_updates[int(y)]+=1
        # clone snapshots: later bags must not mutate tensors needed by backward.
        positive=self.positive_center.detach().clone()
        negative=self.negative_center.detach().clone()
        vectors=output['global_features']
        normalized=F.normalize(vectors,dim=-1)
        gram=normalized@normalized.T+1e-6*torch.eye(len(vectors),device=vectors.device,dtype=vectors.dtype)
        sign,logdet=torch.linalg.slogdet(gram)
        if sign<=0:raise FloatingPointError('Non-positive diversity Gram determinant')
        diversity=-logdet
        alignment=F.triplet_margin_with_distance_loss(vectors,positive.expand_as(vectors),negative.expand_as(vectors),distance_function=lambda a,b:1-F.cosine_similarity(a,b),margin=1.)
        auxiliary=(.1*diversity+.1*alignment) if epoch>=10 else ce.new_zeros(())
        return ce+auxiliary,{'classification':ce,'diversity':diversity,'alignment':alignment,'auxiliary_active':epoch>=10}
