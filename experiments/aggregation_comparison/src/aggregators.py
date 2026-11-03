"""Single unpadded bag [K,D] -> scalar logit. See docs/02_method_adaptations.md."""
import ast
from pathlib import Path
import torch
from torch import nn

ROOT = Path(__file__).resolve().parents[1]
PHPR_SOURCE = ROOT.parents[1] / 'scripts_qc/train_whole_image_effb0_conv_adapter_phpr.py'

def original_phpr_class():
    # Extract only the original class, avoiding the encoder/training entry point.
    tree = ast.parse(PHPR_SOURCE.read_text(encoding='utf-8-sig'))
    node = next(n for n in tree.body if isinstance(n, ast.ClassDef) and n.name == 'PatientPoolingHead')
    scope = {'torch': torch, 'nn': nn}
    exec(compile(ast.Module(body=[node], type_ignores=[]), str(PHPR_SOURCE), 'exec'), scope)
    return scope['PatientPoolingHead']

class Mean(nn.Module):
    def __init__(self, d):
        super().__init__()
        self.classifier = nn.Linear(d, 1)
    def forward(self, x):
        return self.classifier(x.mean(0)).reshape(())

class GatedABMIL(nn.Module):
    def __init__(self, d):
        super().__init__()
        self.v, self.u, self.w = nn.Linear(d,128), nn.Linear(d,128), nn.Linear(128,1)
        self.classifier = nn.Linear(d,1)
    def forward(self, x):
        a = self.w(torch.tanh(self.v(x))*torch.sigmoid(self.u(x))).softmax(0)
        return self.classifier((a*x).sum(0)).reshape(())

class PHPR(nn.Module):
    def __init__(self, d):
        super().__init__()
        self.head = original_phpr_class()(input_dim=d, patient_pooling='phpr', hidden_dim=256,
            dropout=.1, num_prototypes_per_class=1, lambda_proto=.1, lambda_std=0., proto_temperature=.2)
    def forward(self, x):
        return self.head(x, torch.zeros(len(x), device=x.device, dtype=torch.long), 1)[0].reshape(())

AVAILABLE = ('mean','gated_abmil','phpr')
BLOCKED = {'dgr_mil': 'Missing DGR-MIL-main/models/dgrmil.py and original losses/training code',
           'acmil': 'Missing ACMIL-main/architecture/transformer.py and original training code'}

def build(method, d=1280):
    if method in BLOCKED: raise RuntimeError(BLOCKED[method])
    return {'mean': Mean, 'gated_abmil': GatedABMIL, 'phpr': PHPR}[method](d)
