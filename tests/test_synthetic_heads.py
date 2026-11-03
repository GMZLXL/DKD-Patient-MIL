"""Artificial CPU features only. No Dataset construction, images, or checkpoints."""
import sys
import unittest
from pathlib import Path
import torch
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from scripts_qc.train_whole_image_effb0_conv_adapter_phpr import PatientPoolingHead, ConvAdapter

class SyntheticHeadTests(unittest.TestCase):
    def setUp(self):
        torch.set_num_threads(2)
        torch.manual_seed(42)

    def test_variable_bags_finite_and_permutation_invariant(self):
        for pooling in ['mean','gated_attention','phpr']:
            head=PatientPoolingHead(input_dim=32,hidden_dim=16,attention_dim=8,
                patient_pooling=pooling,dropout=0.,num_prototypes_per_class=1).eval()
            for k in [1,2,9]:
                with self.subTest(pooling=pooling,k=k),torch.no_grad():
                    x=torch.randn(k,32)
                    idx=torch.zeros(k,dtype=torch.long)
                    z,features,attention=head(x,idx,1)
                    swapped=head(x.flip(0),idx,1)[0]
                    self.assertEqual(tuple(z.shape),(1,))
                    self.assertTrue(torch.isfinite(z).all())
                    self.assertTrue(torch.isfinite(features).all())
                    self.assertTrue(torch.isfinite(attention[0]).all())
                    self.assertAlmostEqual(float(attention[0].sum()),1.,places=5)
                    torch.testing.assert_close(z,swapped,atol=1e-5,rtol=1e-5)

    def test_unpadded_batch_agrees_with_individual_bags(self):
        for pooling in ['mean','gated_attention','phpr']:
            head=PatientPoolingHead(input_dim=32,hidden_dim=16,attention_dim=8,
                patient_pooling=pooling,dropout=0.,num_prototypes_per_class=1).eval()
            xs=[torch.randn(k,32) for k in [1,2,9]]
            idx=torch.cat([torch.full((len(x),),i,dtype=torch.long) for i,x in enumerate(xs)])
            with torch.no_grad():
                batched=head(torch.cat(xs),idx,3)[0]
                individual=torch.cat([head(x,torch.zeros(len(x),dtype=torch.long),1)[0] for x in xs])
                torch.testing.assert_close(batched,individual,atol=1e-5,rtol=1e-5)

    def test_zero_initialized_conv_adapter_is_identity(self):
        adapter=ConvAdapter(channels=16,dropout=0.).eval()
        x=torch.randn(2,16,8,8)
        with torch.no_grad():torch.testing.assert_close(adapter(x),x,atol=0,rtol=0)

if __name__=='__main__':unittest.main()
