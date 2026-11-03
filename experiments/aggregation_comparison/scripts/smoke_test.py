import json
import sys
from pathlib import Path
import torch
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
from src.aggregators import build, AVAILABLE, BLOCKED
from src.patient_bags import load_cache, split, digest

def main():
    torch.set_num_threads(2)
    torch.manual_seed(42)
    bags,meta = load_cache()
    s = split(bags,0,42)
    result = {'passed':True,'cache_sha256':meta['features_sha256'],'blocked':BLOCKED,'methods':{}}
    for method in AVAILABLE:
        checks = []
        for k in range(1,10):
            model = build(method)
            x = torch.randn(k,1280,requires_grad=True)
            model.train()
            opt = torch.optim.AdamW(model.parameters(),lr=.001)
            z = model(x)
            loss = torch.nn.functional.binary_cross_entropy_with_logits(z,torch.tensor(1.))
            loss.backward()
            assert torch.isfinite(z) and torch.isfinite(x.grad).all()
            assert (x.grad.abs().sum(1)>0).all(), 'An instance has no gradient'
            assert all(torch.isfinite(p.grad).all() for p in model.parameters() if p.grad is not None)
            opt.step()
            model.eval()
            with torch.no_grad():
                torch.testing.assert_close(model(x),model(x[torch.randperm(k)]),atol=2e-5,rtol=2e-5)
            checks.append(k)
        model = build(method)
        opt = torch.optim.AdamW(model.parameters(),lr=.001)
        ids = s['train'][:4]
        before = {n:p.detach().clone() for n,p in model.named_parameters()}
        loss = torch.stack([torch.nn.functional.binary_cross_entropy_with_logits(model(bags[p]['x']),torch.tensor(float(bags[p]['y']))) for p in ids]).mean()
        opt.zero_grad(); loss.backward(); opt.step()
        assert any(not torch.equal(p,before[n]) for n,p in model.named_parameters())
        model.eval()
        tmp = ROOT/'runs'/'smoke'
        tmp.mkdir(parents=True,exist_ok=True)
        torch.save(model.state_dict(),tmp/(method+'.pt'))
        restored = build(method)
        restored.load_state_dict(torch.load(tmp/(method+'.pt'),weights_only=True)); restored.eval()
        with torch.no_grad():
            for pid in ids: torch.testing.assert_close(model(bags[pid]['x']),restored(bags[pid]['x']),atol=0,rtol=0)
        result['methods'][method] = {'passed':True,'K':checks,'real_train_ids':ids,'real_step_loss':float(loss.detach()),'checkpoint_roundtrip':True,'padding':'not used; single unpadded bag only'}
    result['code_sha256']={str(p.relative_to(ROOT)):digest(p) for folder in ['src','scripts'] for p in (ROOT/folder).rglob('*.py')}
    (ROOT/'reports/smoke_test.json').write_text(json.dumps(result,indent=2),encoding='utf-8')
    print(json.dumps(result,indent=2))

if __name__ == '__main__': main()
