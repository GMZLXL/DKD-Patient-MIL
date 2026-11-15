"""Synthetic-only unit tests; no patient files or pretrained checkpoints."""
import argparse
import copy
import json
import sys
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch
import numpy as np
import torch
from torchvision.models import efficientnet_b0

REPO=Path(__file__).resolve().parents[3]
sys.path.insert(0,str(REPO))
import experiments.controlled_pooling_adapter_ema.core as core
import experiments.controlled_pooling_adapter_ema.train as runner

def random_test_encoder(weights):
    with torch.random.fork_rng(devices=[]):
        torch.manual_seed(42)
        backbone=efficientnet_b0(weights=None)
        with patch.object(core.legacy,'efficientnet_b0',return_value=backbone):
            return core.legacy.EfficientNetB0ConvAdapterEncoder()

class ControlledTests(unittest.TestCase):
    def setUp(self):torch.set_num_threads(2);torch.manual_seed(42)

    def test_shared_initialization_and_original_phpr_parameters(self):
        heads=[core.make_head(group) for group in core.GROUPS]
        for head in heads[1:]:
            for name,state in [('proj',heads[0].proj.state_dict()),('classifier',heads[0].classifier.state_dict())]:
                for key,value in state.items():torch.testing.assert_close(value,getattr(head,name).state_dict()[key],atol=0,rtol=0)
        self.assertEqual(tuple(heads[2].prototypes.shape),(2,256))
        self.assertEqual(heads[2].lambda_std,0.)
        self.assertEqual(heads[2].lambda_proto,.1)

    def test_k_1_to_9_image_gradients_permutation_and_mask(self):
        with patch.object(core,'make_encoder',side_effect=random_test_encoder):
            for group in core.GROUPS:
                model=core.ControlledModel(group,'synthetic')
                for k in range(1,10):
                    model.train();model.zero_grad(set_to_none=True)
                    x=torch.randn(k,3,32,32);idx=torch.zeros(k,dtype=torch.long)
                    logits,z,_=model(x,idx,1)
                    self.assertTrue(torch.isfinite(logits).all() and torch.isfinite(z).all())
                    logits.sum().backward()
                    gradient=sum(float(p.grad.abs().sum()) for p in model.encoder.adapters.parameters() if p.grad is not None)
                    self.assertGreater(gradient,0.)
                    self.assertTrue(all(p.grad is None or torch.isfinite(p.grad).all() for p in model.parameters()))
                    self.assertTrue(all(not p.requires_grad for p in model.encoder.features.parameters()))
                    model.eval()
                    padded=torch.cat([x,torch.full((1,3,32,32),float('nan'))])
                    with torch.no_grad():
                        expected=model(x,idx,1)[0]
                        torch.testing.assert_close(expected,model(x.flip(0),idx,1)[0],atol=2e-5,rtol=2e-5)
                        torch.testing.assert_close(expected,model(padded,torch.zeros(k+1,dtype=torch.long),1,
                            torch.tensor([True]*k+[False]))[0],atol=2e-5,rtol=2e-5)
                with self.assertRaises(ValueError):model(x,idx,2)

    def test_loss_matches_original_formula_and_eval_never_updates_ema(self):
        original=core.legacy.WeightedBCEWithEMAPCARLoss(torch.tensor([1.3]))
        wrapped=core.SharedEMAPCAR(torch.tensor([1.3]))
        logits=torch.randn(4,requires_grad=True);z=torch.randn(4,256,requires_grad=True);y=torch.tensor([0.,1.,0.,1.])
        a=original(logits,y,z,update_ema=True);b=wrapped(logits,y,z,phase='train')
        torch.testing.assert_close(a,b,atol=0,rtol=0)
        torch.testing.assert_close(original.ema_prototypes,wrapped.ema_prototypes,atol=0,rtol=0)
        ga=torch.autograd.grad(a,[logits,z],retain_graph=True)
        gb=torch.autograd.grad(b,[logits,z])
        for left,right in zip(ga,gb):torch.testing.assert_close(left,right,atol=0,rtol=0)
        saved=copy.deepcopy(wrapped.state_dict());wrapped.eval()
        for phase in ['val','test']:
            wrapped(logits,y,z,phase=phase)
            for key,value in saved.items():torch.testing.assert_close(value,wrapped.state_dict()[key],atol=0,rtol=0)
        with self.assertRaises(RuntimeError):wrapped(logits,y,z,phase='train')

    def test_empty_and_initialized_ema_roundtrip_and_single_class_loss(self):
        loss=core.SharedEMAPCAR(torch.ones(1))
        restored=core.SharedEMAPCAR(torch.ones(1));restored.load_state_dict(loss.state_dict())
        self.assertFalse(bool(restored.ema_initialized))
        y=torch.ones(4);z=torch.randn(4,256);logits=torch.randn(4)
        value=loss(logits,y,z,phase='train')
        torch.testing.assert_close(value,loss.bce(logits,y))
        restored.load_state_dict(loss.state_dict())
        self.assertTrue(bool(restored.ema_initialized))
        torch.testing.assert_close(restored.calibrate_logits(logits,z),loss.calibrate_logits(logits,z),atol=0,rtol=0)
        fresh=core.SharedEMAPCAR(torch.ones(1));self.assertFalse(bool(fresh.ema_initialized))

    def test_model_and_ema_checkpoint_roundtrip(self):
        with patch.object(core,'make_encoder',side_effect=random_test_encoder):
            for group in core.GROUPS:
                model=core.ControlledModel(group,'synthetic').eval();loss=core.SharedEMAPCAR(torch.ones(1))
                x=torch.randn(2,3,32,32);idx=torch.tensor([0,1]);y=torch.tensor([0.,1.])
                logits,z,_=model(x,idx,2);loss(logits,y,z,phase='train')
                with tempfile.TemporaryDirectory() as tmp:
                    path=Path(tmp)/'state.pt';torch.save(dict(model=model.state_dict(),criterion=loss.state_dict()),path)
                    other=core.ControlledModel(group,'synthetic').eval();other_loss=core.SharedEMAPCAR(torch.ones(1))
                    runner.restore(path,other,other_loss)
                    with torch.no_grad():
                        got,features,_=other(x,idx,2)
                        torch.testing.assert_close(logits,got,atol=0,rtol=0)
                        torch.testing.assert_close(loss.calibrate_logits(logits,z),other_loss.calibrate_logits(got,features),atol=0,rtol=0)

    def test_outer_labels_never_affect_split(self):
        class Outer(dict):
            def __getitem__(self,key):
                if key in {'label','y','paths','images'}:raise AssertionError('Outer data accessed')
                return super().__getitem__(key)
        items={f'synthetic_{i:03d}':dict(fold=i%5,y=i%2) for i in range(205)}
        for fold in range(5):
            a=runner.inner_split_without_test_labels(items,fold,42)
            altered={pid:Outer(value) if value['fold']==fold else value for pid,value in items.items()}
            self.assertEqual(a,runner.inner_split_without_test_labels(altered,fold,42))

    def test_outer_labels_cannot_change_model_ema_checkpoint_or_threshold(self):
        class TinyEncoder(torch.nn.Module):
            def __init__(self):
                super().__init__();self.features=torch.nn.Conv2d(3,16,1)
                self.features.requires_grad_(False)
                self.adapters=torch.nn.ModuleDict({'7':core.legacy.ConvAdapter(16)})
                self.out=torch.nn.Linear(16,1280);self.out.requires_grad_(False)
            def forward(self,x):return self.out(self.adapters['7'](self.features(x)).mean((2,3)))
        class Outer(dict):
            def __getitem__(self,key):
                if key in {'label','y','images','paths'}:raise AssertionError('Outer data accessed')
                return super().__getitem__(key)
        generator=torch.Generator().manual_seed(123)
        items={f'artificial_{i:03d}':dict(patient_id=f'artificial_{i:03d}',fold=i%5,
            y=i%2,label=i%2,images=torch.randn(1,3,8,8,generator=generator),num_images=1) for i in range(205)}
        changed={pid:Outer(dict(value,y=1-value['y'],label=1-value['label'])) if value['fold']==0 else value for pid,value in items.items()}
        def synthetic_loader(data,ids,train,workers):
            self.assertTrue(all(data[p]['fold']!=0 for p in ids))
            records=[dict(data[p],label=torch.tensor(float(data[p]['label']))) for p in ids]
            return torch.utils.data.DataLoader(records,batch_size=4,shuffle=train,
                collate_fn=core.legacy.patient_collate_fn,generator=torch.Generator().manual_seed(42))
        outcomes=[]
        with tempfile.TemporaryDirectory() as tmp:
            artifact=Path(tmp)/'artificial.txt';artifact.write_text('synthetic fixture')
            for i,data in enumerate([items,changed]):
                args=argparse.Namespace(run_root=str(Path(tmp)/str(i)),mode='smoke',weights=str(artifact),
                    index=str(artifact),reference=str(artifact),workers=0)
                with patch.object(core,'make_encoder',side_effect=lambda weights:TinyEncoder()),patch.object(runner,'loader',side_effect=synthetic_loader):
                    report=runner.run_fold(args,data,'C2',0)
                self.assertEqual(report['outer_test_evaluations'],0)
                folder=Path(args.run_root)/'C2/fold_0'
                self.assertFalse((folder/'test_patient_predictions.csv').exists())
                state=torch.load(folder/'best_model.pt',weights_only=True)
                t=json.loads((folder/'val_threshold.json').read_text())['threshold']
                outcomes.append((state,t))
        self.assertEqual(outcomes[0][0]['epoch'],outcomes[1][0]['epoch'])
        self.assertEqual(outcomes[0][1],outcomes[1][1])
        for component in ['model','criterion']:
            for key,value in outcomes[0][0][component].items():
                torch.testing.assert_close(value,outcomes[1][0][component][key],atol=0,rtol=0)

if __name__=='__main__':unittest.main()
