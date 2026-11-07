"""Synthetic tests only: no real medical image or private manifest in Git."""
import copy
import io
import json
from pathlib import Path
import zipfile

import numpy as np
from PIL import Image
import pytest
import torch

from common import CLASSES, metrics_from_probs
from models import OCTCNN, DynamicConv2d, complexity
from noise import normalize_batch, perturb_intensity, make_noise_generator, backward_objective
from prepare_data import prepare, duplicate_rules, source_split_check
from evaluate import freeze_runs

torch.set_num_threads(2)


def test_dynamic_kernel_mixture_matches_individual_convolutions():
    torch.manual_seed(3)
    conv=DynamicConv2d(2,3).double()
    x=torch.randn(3,2,9,11,dtype=torch.double)
    alpha=conv.routing(x)
    expected=torch.cat([torch.nn.functional.conv2d(x[i:i+1],sum(alpha[i,k]*conv.weight[k] for k in range(4)),padding=1) for i in range(3)])
    torch.testing.assert_close(conv(x),expected)


@pytest.mark.parametrize('variant',['basic_noise','na_dynamic'])
def test_streamed_noise_gradient_equals_joint_objective(variant):
    torch.manual_seed(10)
    model=OCTCNN(variant).double()
    reference=copy.deepcopy(model)
    raw=torch.rand(4,1,32,32,dtype=torch.double)
    labels=torch.arange(4)
    ce=torch.nn.CrossEntropyLoss(weight=torch.tensor([1.,2.,3.,1.],dtype=torch.double),label_smoothing=.1)
    cfg={'noise_realizations':4,'noise_sigma':.06,'noise_beta':.12}
    got=backward_objective(model,raw,labels,ce,cfg,variant,make_noise_generator('cpu',42,0,1))
    rng=make_noise_generator('cpu',42,0,1)
    clean=ce(reference(normalize_batch(raw)),labels)
    noisy=[]
    for _ in range(4):
        view=(raw*(1+.06*torch.randn(raw.shape,dtype=raw.dtype,generator=rng))).clamp(0,1)
        view=(view-view.mean((-2,-1),keepdim=True))/view.std((-2,-1),correction=0,keepdim=True).clamp_min(1e-6)
        noisy.append(ce(reference(view),labels))
    objective=clean+.12*torch.stack(noisy).mean()
    objective.backward()
    assert got['total']==pytest.approx(objective.item(),abs=1e-10)
    for a,b in zip(model.parameters(),reference.parameters()):
        torch.testing.assert_close(a.grad,b.grad,rtol=1e-7,atol=1e-9)
    for a,b in zip(model.buffers(),reference.buffers()):
        torch.testing.assert_close(a,b,rtol=0,atol=0)


def test_noise_identity_bounds_and_no_inference_branch():
    raw=torch.rand(2,1,32,32)
    rng=make_noise_generator('cpu',42,0,1)
    assert torch.equal(perturb_intensity(raw,0,rng),raw)
    assert torch.isfinite(normalize_batch(torch.zeros_like(raw))).all()
    for plain,noisy in [('basic','basic_noise'),('dynamic','na_dynamic')]:
        model=OCTCNN(plain).eval();other=OCTCNN(noisy).eval()
        other.load_state_dict(model.state_dict())
        torch.testing.assert_close(model(raw),other(raw),rtol=0,atol=0)
    assert complexity('basic')['parameters']==105908
    assert complexity('dynamic')['parameters']==398168


def test_neh_probability_restriction():
    logits=torch.tensor([[1.,9.,2.,3.]])
    restricted=logits[:,[0,2,3]].softmax(1).numpy()
    assert restricted.sum()==pytest.approx(1)
    result=metrics_from_probs([3],restricted,[0,2,3])
    assert result['accuracy']==1
    assert result['classes']==['CNV','DRUSEN','NORMAL']


def synthetic_archive(path):
    rng=np.random.default_rng(17)
    with zipfile.ZipFile(path,'w') as z:
        for si,split in enumerate(('train','val','test')):
            for ci,label in enumerate(CLASSES):
                for n in range(2):
                    a=rng.integers(0,256,(25,37),dtype=np.uint8)
                    buffer=io.BytesIO();Image.fromarray(a).save(buffer,format='PNG')
                    z.writestr(f'wrapper/{split}/{label}/{label}-{si*100+ci+1}-{n}.png',buffer.getvalue())


def test_zip_smoke_preserves_split_and_never_reads_test(tmp_path,monkeypatch):
    path=tmp_path/'source.zip';synthetic_archive(path)
    original=zipfile.ZipFile.read
    reads=[]
    def checked(self,name,*a,**kw):
        reads.append(name)
        assert '/test/' not in name
        return original(self,name,*a,**kw)
    monkeypatch.setattr(zipfile.ZipFile,'read',checked)
    report=prepare({'OCT2017':path},tmp_path/'audit','smoke',per_class=1)
    assert report['passed'] and len(reads)==8
    assert set(report['manifest_sha256'])=={'OCT2017_train','OCT2017_val'}
    assert np.load(tmp_path/'audit/OCT2017_train.npy').shape==(4,256,256)
    assert not (tmp_path/'audit/OCT2017_test.npy').exists()
    assert prepare({'OCT2017':path},tmp_path/'audit','smoke',per_class=1)==report


def test_patient_overlap_is_rejected():
    rows=[{'dataset':'OCT2017','split':split,'patient_id':'123'} for split in ['train','val']]
    with pytest.raises(ValueError,match='overlap'):source_split_check(rows)


def test_transitive_external_conflict_is_fully_excluded():
    rows=[]
    for dataset,path,label,raw,gray in [('NEH','a','CNV','r1','g1'),('NEH','b','CNV','r1','g2'),('OCT-C8','c','DME','r3','g2')]:
        rows.append(dict(dataset=dataset,path=path,label=label,split='external_test',raw_sha256=raw,pixel_sha256=path,gray_sha256=gray))
    exclusions,errors,groups=duplicate_rules(rows)
    assert not errors and set(exclusions)=={'a','b','c'} and len(groups)==1


def test_formal_freeze_rejects_smoke(tmp_path):
    run=tmp_path/'basic_seed42';run.mkdir()
    (run/'completed.json').write_text(json.dumps({'smoke':True}))
    with pytest.raises(RuntimeError,match='Smoke'):
        freeze_runs(tmp_path,{'variants':['basic'],'seeds':[42]})


def test_notebook_structure_and_cell_syntax():
    path=Path(__file__).resolve().parents[1]/'notebooks/OCT_Colab.ipynb'
    nb=json.loads(path.read_text(encoding='utf8'))
    assert nb['nbformat']==4
    for i,c in enumerate(nb['cells']):
        if c['cell_type']=='code':
            assert c['outputs']==[] and c['execution_count'] is None
            compile(''.join(c['source']),f'cell{i}','exec')
