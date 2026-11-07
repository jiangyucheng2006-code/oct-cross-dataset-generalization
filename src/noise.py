"""JBO Eq.1,4-6: clean CE + beta * mean CE over independent noisy views.

Operate on post-export [0,1] intensities BEFORE per-image normalization.
The noise models processed-image variability, not raw physical OCT speckle.
No noise, averaging, or adaptation is used at validation/test time.
"""
import torch

NOISE_VARIANTS = ('basic_noise', 'na_dynamic')

def normalize_batch(x, eps=1e-6):
    mean=x.mean(dim=(-2,-1),keepdim=True)
    std=x.std(dim=(-2,-1),correction=0,keepdim=True).clamp_min(eps)
    return (x-mean)/std

def perturb_intensity(x, sigma, generator):
    noise=torch.randn(x.shape,dtype=x.dtype,device=x.device,generator=generator)
    return (x*(1+sigma*noise)).clamp(0,1)

def make_noise_generator(device, seed, epoch, step):
    # Dedicated stream: noise never changes batch order, model initialization,
    # or mild spatial/photometric augmentation; paired across both backbones.
    value=(seed*1000000007+epoch*1000003+step*97+71237) % (2**63-1)
    return torch.Generator(device=device).manual_seed(value)

def backward_objective(model, raw, targets, criterion, cfg, variant, generator=None):
    """Accumulate gradients over views, then caller performs ONE optimizer step.

    Sequential forwards retain the same graph/BN-update order as a direct
    clean-plus-noisy sum, while retaining only one view's graph at a time.
    The augmented mean is weighted by beta/N, NOT beta per realization.
    """
    clean=criterion(model(normalize_batch(raw)),targets)
    if not torch.isfinite(clean):raise RuntimeError('Non-finite clean loss')
    clean.backward()
    augmented=0.0
    if variant in NOISE_VARIANTS:
        n=int(cfg['noise_realizations']);beta=float(cfg['noise_beta'])
        if n<1 or beta<0 or generator is None:raise ValueError('Invalid noise objective')
        for _ in range(n):
            perturbed=perturb_intensity(raw,float(cfg['noise_sigma']),generator)
            loss=criterion(model(normalize_batch(perturbed)),targets)
            if not torch.isfinite(loss):raise RuntimeError('Non-finite noise loss')
            (loss*(beta/n)).backward()
            augmented+=float(loss.detach())/n
    clean_value=float(clean.detach())
    return {'clean':clean_value,'noise':augmented,
            'total':clean_value+(cfg['noise_beta']*augmented if variant in NOISE_VARIANTS else 0.0)}
