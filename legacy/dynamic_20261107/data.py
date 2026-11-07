import csv
import random
from pathlib import Path
import numpy as np
import torch
from PIL import Image, ImageEnhance, ImageOps
from torch.utils.data import Dataset

class OCTDataset(Dataset):
    def __init__(self,audit_dir,key,training=False,seed=42,limit=None):
        if training and key!='OCT2017_train':raise ValueError('Only OCT2017_train may update weights')
        self.root=Path(audit_dir);self.key=key;self.training=training;self.seed=seed;self.epoch=0
        with (self.root/(key+'.csv')).open() as f:self.rows=list(csv.DictReader(f))
        if limit:self.rows=self.rows[:limit]
        self.images=None
        self.labels=np.asarray([int(r['class_index']) for r in self.rows],dtype=np.int64)

    def __len__(self):return len(self.rows)

    def __getitem__(self,index):
        if self.images is None:self.images=np.load(self.root/(self.key+'.npy'),mmap_mode='r')
        r=self.rows[index];image=Image.fromarray(np.asarray(self.images[int(r['cache_index'])]))
        if self.training:
            # Stateless augmentation: same example/epoch/seed across all variants;
            # independent of DataLoader workers and exact checkpoint resumes.
            rng=random.Random(self.seed*1000000007+self.epoch*1000003+int(r['cache_index']))
            if rng.random()<.5:image=ImageOps.mirror(image)
            image=image.rotate(rng.uniform(-5,5),resample=Image.Resampling.BILINEAR,fillcolor=0)
            scale=rng.uniform(.95,1.05);size=image.width
            tx=rng.uniform(-.03,.03)*size;ty=rng.uniform(-.03,.03)*size
            inv=1/scale;offset=(size-1)/2*(1-inv)
            image=image.transform(image.size,Image.Transform.AFFINE,(inv,0,offset-tx*inv,0,inv,offset-ty*inv),
                                  resample=Image.Resampling.BILINEAR,fillcolor=0)
            image=ImageEnhance.Brightness(image).enhance(rng.uniform(.9,1.1))
            image=ImageEnhance.Contrast(image).enhance(rng.uniform(.9,1.1))
        a=np.asarray(image,dtype=np.float32)/255.0
        a=(a-a.mean())/max(float(a.std()),1e-6)
        return torch.from_numpy(a[None].copy()),int(r['class_index']),index
