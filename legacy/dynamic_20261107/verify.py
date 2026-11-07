"""Numerical tests for dynamic convolution, gradients, metrics and class mapping."""
import json, tempfile, time
from pathlib import Path
import numpy as np
import torch
from torch.nn import functional as F
from common import metrics_from_probs, NEH_INDICES, write_json
from models import DynamicConv2d, OCTCNN, complexity
from train import seed_everything

def main():
    seed_everything(42);torch.set_num_threads(2)
    m=DynamicConv2d(2,3).double();x=torch.randn(2,2,6,6,dtype=torch.double,requires_grad=True)
    actual=m(x)
    alpha=m.routing(x)
    expected=torch.cat([F.conv2d(x[i:i+1],(alpha[i,:,None,None,None,None]*m.weight).sum(0),padding=1) for i in range(len(x))])
    torch.testing.assert_close(actual,expected,rtol=1e-9,atol=1e-10)
    variables=[x,*m.parameters()];upstream=torch.randn_like(actual)
    g1=torch.autograd.grad((actual*upstream).sum(),variables,retain_graph=True)
    g2=torch.autograd.grad((expected*upstream).sum(),variables)
    for a,b in zip(g1,g2):torch.testing.assert_close(a,b,rtol=1e-8,atol=1e-9)
    for stats in [False,True]:
        a=DynamicConv2d(2,3,router_stats=stats)(torch.randn(3,2,8,8))
        assert a.shape==(3,3,8,8) and torch.isfinite(a).all()
    toy=torch.tensor([[.1,.7,.15,.05]],dtype=torch.float64)
    restricted=toy.log()[:,NEH_INDICES].softmax(1)
    torch.testing.assert_close(restricted,torch.tensor([[1/3,1/2,1/6]],dtype=torch.float64))
    score=metrics_from_probs([2],restricted.numpy(),NEH_INDICES)
    assert score['accuracy']==1 and score['classes']==['CNV','DRUSEN','NORMAL']
    perfect=metrics_from_probs([0,1,2,3],np.eye(4))
    assert perfect['accuracy']==perfect['macro_f1']==perfect['balanced_accuracy']==1
    assert perfect['ece_15_equal_mass']==0
    device='cuda';assert torch.cuda.is_available();benchmarks={}
    for variant in ['basic','dynamic','dynamic_stats','dynamic_balance']:
        model=OCTCNN(variant).to(device);model.set_epoch(10)
        xb=torch.randn(8,1,64,64,device=device);yb=torch.arange(8,device=device)%4
        opt=torch.optim.AdamW(model.parameters(),lr=.001);losses=[]
        for _ in range(8):
            model.train();opt.zero_grad();out=model(xb);loss=F.cross_entropy(out,yb)
            if variant=='dynamic_balance':loss=loss+.01*model.routing_balance_loss()
            loss.backward();assert all(torch.isfinite(p.grad).all() for p in model.parameters() if p.grad is not None)
            opt.step();losses.append(float(loss.detach()))
        assert losses[-1]<losses[0],(variant,losses)
        model.eval()
        with torch.inference_mode():before=model(xb)
        with tempfile.TemporaryDirectory() as tmp:
            f=Path(tmp)/'state.pt';torch.save(model.state_dict(),f)
            clone=OCTCNN(variant).to(device);clone.load_state_dict(torch.load(f,weights_only=True));clone.set_epoch(10);clone.eval()
            with torch.inference_mode():torch.testing.assert_close(before,clone(xb),rtol=0,atol=0)
        batch=torch.randn(64,1,256,256,device=device);target=torch.arange(64,device=device)%4
        times=[];model.train()
        for repeat in range(4):
            torch.cuda.synchronize();t=time.monotonic();opt.zero_grad();F.cross_entropy(model(batch),target).backward();opt.step()
            torch.cuda.synchronize()
            if repeat:times.append(time.monotonic()-t)
        benchmarks[variant]={'seconds_per_train_batch64':float(np.median(times)),'tiny_loss_first_last':[losses[0],losses[-1]],**complexity(variant)}
    result={'passed':True,'torch':torch.__version__,'gpu':torch.cuda.get_device_name(),'tests':['per-sample forward equivalence','all parameter and input gradient equivalence','three-class renormalization','class ordering and perfect metrics','GPU training loss decrease','checkpoint reload exact predictions'],'benchmarks':benchmarks}
    write_json('verification.json',result);print(json.dumps(result,indent=2),flush=True)

if __name__=='__main__':main()
