"""Independent paper-based reconstruction; not the authors' released code.

Chutani & Yalavarthy, JBO 2026, DOI 10.1117/1.JBO.31.8.086006.
Dynamic kernel mixing follows Chen et al., CVPR 2020 (1912.03458).
Unspecified layer details are explicitly listed in RECONSTRUCTION.md.
"""
import math
import torch
from torch import nn
from torch.nn import functional as F

class DynamicConv2d(nn.Module):
    def __init__(self, in_channels, out_channels, experts=4, router_stats=False):
        super().__init__()
        self.in_channels=in_channels;self.out_channels=out_channels;self.experts=experts
        self.router_stats=router_stats;self.temperature=1.0;self.last_routing=None
        self.weight=nn.Parameter(torch.empty(experts,out_channels,in_channels,3,3))
        self.router=nn.Sequential(nn.Linear(in_channels*(2 if router_stats else 1),max(4,in_channels//4)),
                                  nn.ReLU(),nn.Linear(max(4,in_channels//4),experts))
        for w in self.weight:nn.init.kaiming_normal_(w,mode='fan_out',nonlinearity='relu')

    def routing(self,x):
        descriptor=x.mean(dim=(-2,-1))
        if self.router_stats:
            std=torch.sqrt(x.var(dim=(-2,-1),correction=0)+1e-6)
            descriptor=torch.cat((descriptor,std),1)
        return F.softmax(self.router(descriptor)/self.temperature,dim=1)

    def forward(self,x):
        b,c,h,w=x.shape
        alpha=self.routing(x);self.last_routing=alpha
        weight=(alpha@self.weight.flatten(1)).reshape(b*self.out_channels,c,3,3)
        # Each sample has its own effective kernel; groups isolate samples.
        y=F.conv2d(x.reshape(1,b*c,h,w),weight,padding=1,groups=b)
        return y.reshape(b,self.out_channels,h,w)

class OCTCNN(nn.Module):
    def __init__(self,variant='dynamic'):
        super().__init__();self.variant=variant
        if variant not in ('basic','basic_noise','dynamic','na_dynamic','dynamic_stats','dynamic_balance'):
            raise ValueError(variant)
        self.blocks=nn.ModuleList();cin=1
        for cout in [16,32,64,128]:
            conv=nn.Conv2d(cin,cout,3,padding=1,bias=False) if variant in ('basic','basic_noise') else DynamicConv2d(cin,cout,router_stats=(variant=='dynamic_stats'))
            self.blocks.append(nn.Sequential(conv,nn.BatchNorm2d(cout),nn.ReLU(),nn.MaxPool2d(2)))
            cin=cout
        self.head=nn.Sequential(nn.Linear(128,64),nn.ReLU(),nn.Linear(64,4))

    def set_epoch(self,epoch):
        # Original Dynamic Convolution paper's warm routing, not specified in JBO.
        tau=30.0-29.0*min(epoch,9)/9.0
        for module in self.modules():
            if isinstance(module,DynamicConv2d):module.temperature=tau

    def forward(self,x):
        for block in self.blocks:x=block(x)
        return self.head(x.mean(dim=(-2,-1)))

    def routing_balance_loss(self):
        # Proposed source-only router regularizer, not a reported JBO method.
        losses=[]
        for module in self.modules():
            if isinstance(module,DynamicConv2d) and module.last_routing is not None:
                p=module.last_routing.mean(0)
                losses.append((p*(torch.log(p.clamp_min(1e-8))+math.log(len(p)))).sum())
        return torch.stack(losses).mean() if losses else next(self.parameters()).sum()*0

def complexity(variant,size=256):
    """Analytical MAC estimate with explicit conventions, not a FLOPs profiler."""
    model=OCTCNN(variant);params=sum(p.numel() for p in model.parameters())
    conv_macs=0;router_macs=0;mix_macs=0;cin=1;s=size
    for cout in [16,32,64,128]:
        conv_macs+=s*s*cout*cin*9
        if variant not in ('basic','basic_noise'):
            hidden=max(4,cin//4);descriptor=cin*(2 if variant=='dynamic_stats' else 1)
            router_macs+=descriptor*hidden+hidden*4
            mix_macs+=4*cout*cin*9
        s//=2;cin=cout
    head_macs=128*64+64*4
    total=conv_macs+router_macs+mix_macs+head_macs
    return {'parameters':params,'macs_per_image_batch1':total,'arithmetic_flops_2_per_mac':2*total,
            'included':['convolutions','router linear layers','kernel mixtures','classifier linear layers'],
            'excluded':['pooling','normalization','activation','softmax','router mean and variance reductions'],
            'not_exact_published_flops':True}
