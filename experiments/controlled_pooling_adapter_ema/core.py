"""Reuse the historical innovation modules; no frozen-feature training path."""
import hashlib
from pathlib import Path
from unittest.mock import patch
import torch
from torch import nn
from torchvision.models import efficientnet_b0
import scripts_qc.train_effb0_conv_adapter_phpr_ema_pcar_auc as legacy

GROUPS={'C0':'mean','C1':'gated_attention','C2':'phpr'}
FIXED=dict(image_size=384,patient_batch_size=4,num_workers=4,num_prototypes_per_class=1,
 lambda_proto=.1,lambda_std=0.,proto_temperature=.2,loss_mode='ema_pcar_auc',
 auc_loss_weight=.02,auc_margin=.2,pcar_beta=.03,pcar_temperature=1.,ema_momentum=.9,
 adapter_indices='7,8',adapter_reduction=8,adapter_alpha=.1,adapter_dropout=.1,
 hidden_dim=256,attention_dim=128,head_dropout=.1,epochs=100,patience=50,
 adapter_lr=1e-5,head_lr=1e-4,weight_decay=.0001,grad_clip=5.,unfreeze_bn=False,
 no_amp=False,seed=42,amp=True)

def sha(path):
    h=hashlib.sha256()
    with Path(path).open('rb') as f:
        for block in iter(lambda:f.read(1024*1024),b''):h.update(block)
    return h.hexdigest()

def validate_reference(args):
    for key,value in FIXED.items():
        if args.get(key)!=value:raise ValueError('Historical reference mismatch: '+key)
    if args.get('patient_pooling')!='phpr' or args.get('pcar_eval_score')!='calibrated':
        raise ValueError('Not the user-designated historical full-model reference')

def make_head(group):
    if group not in GROUPS:raise ValueError('Only core groups C0/C1/C2 are implemented')
    def head(mode):
        return legacy.PatientPoolingHead(input_dim=1280,hidden_dim=256,attention_dim=128,
            dropout=.1,patient_pooling=mode,num_prototypes_per_class=1,
            lambda_proto=.1,lambda_std=0.,proto_temperature=.2)
    # Identical common projection/classifier values despite variant-specific RNG consumption.
    with torch.random.fork_rng(devices=[]):
        torch.manual_seed(42);common=head('mean')
        torch.manual_seed(42);selected=head(GROUPS[group])
    selected.proj.load_state_dict(common.proj.state_dict())
    selected.classifier.load_state_dict(common.classifier.state_dict())
    return selected

def make_encoder(weights):
    if sha(weights)!='7f5810bc96def8f7552d5b7e68d53c4786f81167d28291b21c0d90e1fca14934':
        raise ValueError('Wrong ImageNet B0 weights')
    with torch.random.fork_rng(devices=[]):
        torch.manual_seed(42)
        backbone=efficientnet_b0(weights=None)
        backbone.load_state_dict(torch.load(weights,map_location='cpu',weights_only=True),strict=True)
        # Offline constructor dependency injection; original encoder body is unchanged.
        with patch.object(legacy,'efficientnet_b0',return_value=backbone):
            encoder=legacy.EfficientNetB0ConvAdapterEncoder(adapter_indices='7,8',
                adapter_reduction=8,adapter_alpha=.1,adapter_dropout=.1,freeze_bn=True)
    return encoder

class ControlledModel(nn.Module):
    def __init__(self,group,weights):
        super().__init__()
        self.encoder=make_encoder(weights)
        self.patient_head=make_head(group)
        self.group=group

    def forward(self,images,patient_indices,num_patients,valid_mask=None):
        if valid_mask is not None:
            if valid_mask.dtype!=torch.bool or valid_mask.shape!=(len(images),):raise ValueError('Invalid mask')
            images=images[valid_mask];patient_indices=patient_indices[valid_mask]
        if images.ndim!=4 or patient_indices.shape!=(len(images),):raise ValueError('Expected unpadded image batch')
        if set(patient_indices.detach().cpu().tolist())!=set(range(num_patients)):
            raise ValueError('Empty/invalid patient bag')
        features=self.encoder(images)  # Deliberately NOT inside no_grad or detached.
        return self.patient_head(features,patient_indices,num_patients)

class SharedEMAPCAR(legacy.WeightedBCEWithEMAPCARLoss):
    """Same formula; phase enforcement and dynamic EMA-buffer restoration only."""
    def __init__(self,pos_weight):
        super().__init__(pos_weight=pos_weight,auc_loss_weight=.02,auc_margin=.2,
            pcar_beta=.03,pcar_temperature=1.,ema_momentum=.9)

    def forward(self,logits,labels,patient_features,phase):
        if phase not in {'train','val','test'}:raise ValueError('Unknown phase')
        if phase=='train' and not self.training:raise RuntimeError('Training EMA update in eval mode')
        return super().forward(logits,labels,patient_features,update_ema=phase=='train')

    def load_state_dict(self,state_dict,strict=True):
        if 'ema_prototypes' in state_dict:
            saved=state_dict['ema_prototypes']
            self.ema_prototypes=torch.empty(saved.shape,device=self.bce.pos_weight.device,dtype=saved.dtype)
        return super().load_state_dict(state_dict,strict=strict)

def make_optimizer(model):
    return legacy.make_optimizer(model,FIXED['adapter_lr'],FIXED['head_lr'],FIXED['weight_decay'])

def make_scheduler(optimizer):
    return torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer,mode='max',factor=.5,patience=5)

def counts(model):
    return dict(total=sum(p.numel() for p in model.parameters()),
        trainable=sum(p.numel() for p in model.parameters() if p.requires_grad),
        encoder_trainable=sum(p.numel() for p in model.encoder.parameters() if p.requires_grad),
        head_trainable=sum(p.numel() for p in model.patient_head.parameters() if p.requires_grad))
