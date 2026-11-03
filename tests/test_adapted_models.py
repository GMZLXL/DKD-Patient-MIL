import sys
import unittest
from pathlib import Path
import torch
ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/'experiments/aggregation_comparison'))
from src.adapted_models.dgr_mil import DGRMILPatient
from src.adapted_models.acmil import ACMILPatient
from src.adapted_models.nystrom import MaskedNystromAttention,masked_softmax

class AdaptedModelTests(unittest.TestCase):
    def setUp(self):
        torch.set_num_threads(2);torch.manual_seed(42)

    def test_k_1_through_9_forward_backward_and_permutation(self):
        for cls in [DGRMILPatient,ACMILPatient]:
            model=cls(input_dim=32,hidden_dim=32)
            opt=torch.optim.AdamW(model.parameters(),lr=.001)
            for k in range(1,10):
                with self.subTest(model=cls.__name__,k=k):
                    x=torch.randn(k,32,requires_grad=True);model.train();opt.zero_grad()
                    output=model(x)
                    loss,parts=model.objective(output,k%2,epoch=10)
                    loss.backward()
                    self.assertTrue(torch.isfinite(loss))
                    self.assertTrue(torch.isfinite(x.grad).all())
                    self.assertTrue(all(p.grad is None or torch.isfinite(p.grad).all() for p in model.parameters()))
                    before=model.classifier.weight.detach().clone() if cls==DGRMILPatient else model.bag_head.weight.detach().clone()
                    opt.step()
                    after=model.classifier.weight if cls==DGRMILPatient else model.bag_head.weight
                    self.assertFalse(torch.equal(before,after))
                    model.eval();x=torch.randn(k,32,requires_grad=True)
                    predicted=model(x)['logits'];predicted[1].backward()
                    self.assertTrue((x.grad.abs().sum(1)>0).all(),'Every unmasked instance must influence inference')
                    with torch.no_grad():torch.testing.assert_close(predicted,model(x.flip(0))['logits'],atol=2e-5,rtol=2e-5)

    def test_padding_is_ignored_and_has_zero_gradient(self):
        for cls in [DGRMILPatient,ACMILPatient]:
            model=cls(input_dim=32,hidden_dim=32).eval()
            for k in range(1,10):
                x=torch.randn(k,32)
                padded=torch.cat([torch.full((2,32),float('nan')),x,torch.full((3,32),1e10)]).requires_grad_()
                mask=torch.tensor([False]*2+[True]*k+[False]*3)
                output=model(padded,mask)['logits']
                torch.testing.assert_close(output,model(x)['logits'],atol=2e-5,rtol=2e-5)
                output[1].backward()
                self.assertTrue(torch.isfinite(padded.grad).all())
                self.assertTrue((padded.grad[~mask]==0).all())
            with self.assertRaises(ValueError):model(torch.zeros(3,32),torch.zeros(3,dtype=torch.bool))

    def test_acmil_never_masks_all_and_eval_is_deterministic(self):
        model=ACMILPatient(input_dim=32,hidden_dim=32,mask_drop=1.)
        for k in range(1,10):
            model.train();output=model(torch.randn(k,32))
            self.assertEqual(output['candidate_count'],min(10,k-1))
            self.assertEqual(output['masked_count'],k-1)
            self.assertTrue(output['keep_mask'].any(-1).all())
            self.assertTrue((output['attention'][~output['keep_mask']]==0).all())
            torch.testing.assert_close(output['attention'].sum(-1),torch.ones(5))
        model.eval();x=torch.randn(9,32)
        a=model(x);b=model(x)
        self.assertEqual(a['masked_count'],0)
        torch.testing.assert_close(a['logits'],b['logits'],atol=0,rtol=0)

    def test_dgr_centers_do_not_update_in_inference(self):
        model=DGRMILPatient(input_dim=32,hidden_dim=32)
        before={n:b.clone() for n,b in model.named_buffers()}
        model.eval();model(torch.randn(9,32))
        self.assertTrue(all(torch.equal(before[n],b) for n,b in model.named_buffers()))
        with self.assertRaises(RuntimeError):model.objective(model(torch.randn(2,32)),1,epoch=10)
        model.train();model.objective(model(torch.randn(2,32)),1,epoch=10)
        self.assertEqual(model.center_updates.tolist(),[0,1])
        torch.testing.assert_close(model.negative_center,before['negative_center'])

    def test_nystrom_internal_padding_masks_and_finite_gradients(self):
        # Force the approximate path and empty landmark groups; K=1..9 in the
        # production models instead uses their exact path (landmarks=128).
        layer=MaskedNystromAttention(32,heads=4,landmarks=8,dropout=0.,residual_kind='identity')
        x=torch.randn(1,13,32,requires_grad=True)
        output=layer(x);output.square().sum().backward()
        self.assertTrue(torch.isfinite(output).all() and torch.isfinite(x.grad).all())
        self.assertEqual(layer.last_diagnostics[0]['internal_padding'],3)
        self.assertEqual(layer.last_diagnostics[0]['mode'],'masked_nystrom')
        padded=torch.cat([x.detach(),torch.ones(1,4,32)*1e9],dim=1)
        mask=torch.tensor([[True]*13+[False]*4])
        torch.testing.assert_close(layer(padded,mask)[:,:13],output,atol=2e-5,rtol=2e-5)
        self.assertTrue((layer(padded,mask)[:,13:]==0).all())
        zeros=masked_softmax(torch.randn(2,3),torch.zeros(2,3,dtype=torch.bool))
        self.assertTrue((zeros==0).all())

if __name__=='__main__':unittest.main()
