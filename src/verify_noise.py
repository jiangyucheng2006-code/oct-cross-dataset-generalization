"""Numerical and GPU tests of the reconstructed paper noise objective."""
import copy
import json
import time
import numpy as np
import torch
from torch import nn
from common import write_json
from models import OCTCNN, DynamicConv2d, complexity
from noise import normalize_batch, perturb_intensity, make_noise_generator, backward_objective
from train import seed_everything, verify_audit
from data import OCTDataset

def main():
    cfg=json.loads(open('config.json').read());seed_everything(42);torch.set_num_threads(2)
    verify_audit(cfg['audit_dir']);assert torch.cuda.is_available()
    x=torch.linspace(0,1,2*16*16,dtype=torch.double).reshape(2,1,16,16)
    g=torch.Generator().manual_seed(8)
    got=perturb_intensity(x,.06,g)
    expected=(x*(1+.06*torch.randn(x.shape,dtype=x.dtype,generator=torch.Generator().manual_seed(8)))).clamp(0,1)
    torch.testing.assert_close(got,expected,rtol=0,atol=0)
    assert got.min()>=0 and got.max()<=1 and got[0,0,0,0]==0
    torch.testing.assert_close(perturb_intensity(x,0,g),x,rtol=0,atol=0)
    assert not torch.equal(got,perturb_intensity(x,.06,g))
    normalized=normalize_batch(x)
    torch.testing.assert_close(normalized.mean((-2,-1)),torch.zeros(2,1,dtype=x.dtype),atol=1e-12,rtol=0)
    torch.testing.assert_close(normalized.std((-2,-1),correction=0),torch.ones(2,1,dtype=x.dtype))
    assert torch.equal(normalize_batch(torch.zeros_like(x)),torch.zeros_like(x))
    assert torch.equal(make_noise_generator('cpu',42,1,2).get_state(),make_noise_generator('cpu',42,1,2).get_state())

    # Same raw images, transforms and labels in both conditions; normalization
    # occurs AFTER perturbation. Check actual cached training examples remotely.
    raw_ds=OCTDataset(cfg['audit_dir'],'OCT2017_train',True,42,normalize=False)
    norm_ds=OCTDataset(cfg['audit_dir'],'OCT2017_train',True,42)
    for idx in [0,len(raw_ds)//2,len(raw_ds)-1]:
        raw,y,i=raw_ds[idx];norm,y2,i2=norm_ds[idx]
        assert raw.min()>=0 and raw.max()<=1 and (y,i)==(y2,i2)
        torch.testing.assert_close(normalize_batch(raw[None])[0],norm,rtol=1e-5,atol=2e-6)

    # Independently written joint objective must have the same gradients as
    # streamed backward. Includes weighted/smoothed CE and BN running updates.
    for variant in ['basic_noise','na_dynamic']:
        model=OCTCNN(variant).double();model.set_epoch(10);reference=copy.deepcopy(model)
        raw=torch.rand(4,1,32,32,dtype=torch.double);targets=torch.arange(4)
        criterion=nn.CrossEntropyLoss(weight=torch.tensor([1.,2.,3.,1.],dtype=torch.double),label_smoothing=.1)
        model.train();reference.train()
        values=backward_objective(model,raw,targets,criterion,cfg,variant,make_noise_generator('cpu',43,4,2))
        generator=make_noise_generator('cpu',43,4,2)
        clean=criterion(reference(normalize_batch(raw)),targets)
        noises=[]
        for _ in range(cfg['noise_realizations']):
            eta=torch.randn(raw.shape,dtype=raw.dtype,generator=generator)*cfg['noise_sigma']
            nx=(raw*(1+eta)).clamp(0,1)
            nx=(nx-nx.mean((-2,-1),keepdim=True))/nx.std((-2,-1),correction=0,keepdim=True).clamp_min(1e-6)
            noises.append(criterion(reference(nx),targets))
        expected_loss=clean+cfg['noise_beta']*torch.stack(noises).mean()
        expected_loss.backward()
        assert abs(values['total']-float(expected_loss.detach()))<1e-10
        for (name,a),(name2,b) in zip(model.named_parameters(),reference.named_parameters()):
            assert name==name2
            torch.testing.assert_close(a.grad,b.grad,rtol=1e-7,atol=1e-9,msg=lambda msg:name+': '+msg)
        for a,b in zip(model.buffers(),reference.buffers()):torch.testing.assert_close(a,b,rtol=0,atol=0)
        if variant=='na_dynamic':
            grads=[p.grad.abs().sum().item() for n,p in model.named_parameters() if 'router' in n]
            assert sum(grads)>0,'Noise objective must backpropagate through shared routers'

    benchmarks={}
    for variant in ['basic','dynamic','basic_noise','na_dynamic']:
        seed_everything(42);model=OCTCNN(variant).cuda();model.set_epoch(10)
        raw=torch.rand(8,1,64,64,device='cuda');y=torch.arange(8,device='cuda')%4
        criterion=nn.CrossEntropyLoss(label_smoothing=.1);opt=torch.optim.AdamW(model.parameters(),lr=.001)
        losses=[]
        for step in range(8):
            model.train();opt.zero_grad(set_to_none=True)
            v=backward_objective(model,raw,y,criterion,cfg,variant,make_noise_generator('cuda',42,0,step))
            assert all(torch.isfinite(p.grad).all() for p in model.parameters() if p.grad is not None)
            opt.step();losses.append(v['clean'])
        assert losses[-1]<losses[0],(variant,losses)
        # Eval reconstruction has the exact same architecture and no noise branch.
        model.eval();base=OCTCNN('basic' if variant.startswith('basic') else 'dynamic').cuda()
        base.load_state_dict(model.state_dict());base.set_epoch(10);base.eval()
        with torch.inference_mode():
            torch.testing.assert_close(model(normalize_batch(raw)),base(normalize_batch(raw)),rtol=0,atol=0)
        batch=torch.rand(64,1,256,256,device='cuda');target=torch.arange(64,device='cuda')%4
        times=[];model.train();torch.cuda.reset_peak_memory_stats()
        for step in range(3):
            torch.cuda.synchronize();t=time.monotonic();opt.zero_grad(set_to_none=True)
            backward_objective(model,batch,target,criterion,cfg,variant,make_noise_generator('cuda',42,1,step))
            opt.step();torch.cuda.synchronize()
            if step:times.append(time.monotonic()-t)
        benchmarks[variant]={'seconds_per_batch64':float(np.median(times)),
                             'peak_cuda_memory':torch.cuda.max_memory_allocated(),
                             'tiny_clean_loss_first_last':[losses[0],losses[-1]],**complexity(variant)}
    assert benchmarks['basic']['parameters']==benchmarks['basic_noise']['parameters']==105908
    assert benchmarks['dynamic']['parameters']==benchmarks['na_dynamic']['parameters']==398168
    result={'passed':True,'gpu':torch.cuda.get_device_name(),'torch':torch.__version__,
            'tests':['exact multiplicative/clipped noise formula','independent reproducible views',
                     'zero-noise identity and stable normalization','cached raw/normalized input parity',
                     'all gradients equal independent clean+beta*mean(noisy) objective',
                     'shared router gradients','GPU finite gradients and tiny loss decrease',
                     'identical inference for matching state dictionaries','full-size batch64 fits'],
            'benchmarks':benchmarks}
    write_json('verification_noise.json',result);print(json.dumps(result,indent=2),flush=True)

if __name__=='__main__':main()
