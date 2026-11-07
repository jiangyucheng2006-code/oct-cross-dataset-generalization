import argparse, csv, hashlib, json, math, os, platform, random, shutil, subprocess, time
from pathlib import Path
os.environ.setdefault('CUBLAS_WORKSPACE_CONFIG', ':4096:8')
import numpy as np
import torch
from torch.utils.data import DataLoader
from common import CLASSES, write_json, sha256_file, metrics_from_probs
from data import OCTDataset
from models import OCTCNN, complexity
from noise import backward_objective, make_noise_generator

def seed_everything(seed):
    random.seed(seed);np.random.seed(seed);torch.manual_seed(seed);torch.cuda.manual_seed_all(seed)
    torch.use_deterministic_algorithms(True)
    torch.backends.cudnn.benchmark=False;torch.backends.cudnn.deterministic=True
    torch.backends.cuda.matmul.allow_tf32=False;torch.backends.cudnn.allow_tf32=False

def atomic_save(obj,path):
    path=Path(path);tmp=path.with_suffix('.tmp');torch.save(obj,tmp);tmp.replace(path)

def sync_small(out,archive):
    if not archive:return
    target=Path(archive)/Path(out).name;target.mkdir(parents=True,exist_ok=True)
    for p in Path(out).iterdir():
        if p.is_file() and p.suffix in ('.json','.jsonl','.pt','.csv'):
            tmp=target/(p.name+'.tmp');shutil.copyfile(p,tmp);tmp.replace(target/p.name)

@torch.inference_mode()
def validate(model,loader,device,prediction_path=None):
    model.eval();labels=[];probs=[];row_indices=[]
    for x,y,indices in loader:
        logits=model(x.to(device,non_blocking=True))
        probs.append(logits.softmax(1).cpu().numpy());labels.append(y.numpy());row_indices.extend(indices.tolist())
    labels=np.concatenate(labels);probs=np.concatenate(probs)
    if prediction_path:
        with Path(prediction_path).open('w',newline='') as f:
            writer=csv.writer(f);writer.writerow(['path','patient_id','label','prediction',*['p_'+c for c in CLASSES]])
            for i,index in enumerate(row_indices):
                row=loader.dataset.rows[index]
                writer.writerow([row['path'],row['patient_id'],CLASSES[labels[i]],CLASSES[probs[i].argmax()],*probs[i].tolist()])
    return metrics_from_probs(labels,probs)

def verify_audit(audit_dir):
    root=Path(audit_dir);audit=json.loads((root/'audit.json').read_text())
    if not audit['passed']:raise RuntimeError('Data audit did not pass')
    for key in ['OCT2017_train','OCT2017_val']:
        if sha256_file(root/(key+'.csv'))!=audit['manifest_sha256'][key]:raise RuntimeError('Manifest changed: '+key)
    return audit

def main():
    p=argparse.ArgumentParser();p.add_argument('--config',required=True);p.add_argument('--variant',required=True)
    p.add_argument('--seed',type=int,required=True);p.add_argument('--out',required=True);p.add_argument('--archive')
    p.add_argument('--resume',action='store_true');p.add_argument('--smoke',action='store_true')
    p.add_argument('--device', choices=['auto','cpu','cuda'], default='auto')
    args=p.parse_args();cfg=json.loads(Path(args.config).read_text());cfg.update(variant=args.variant,seed=args.seed,smoke=args.smoke)
    cfg_hash=hashlib.sha256(json.dumps(cfg,sort_keys=True).encode()).hexdigest()
    out=Path(args.out);out.mkdir(parents=True,exist_ok=True)
    if (out/'completed.json').exists():raise RuntimeError('Completed run is immutable; choose a new run directory')
    if (out/'last.pt').exists() and not args.resume:raise RuntimeError('Checkpoint exists: explicitly resume')
    audit=verify_audit(cfg['audit_dir'])
    if audit.get('preparation_mode')=='smoke' and not args.smoke:raise RuntimeError('Smoke cache cannot be used for formal training')
    seed_everything(args.seed)
    torch.set_num_threads(2)
    device=torch.device(('cuda' if torch.cuda.is_available() else 'cpu') if args.device=='auto' else args.device)
    if device.type=='cpu' and not args.smoke:raise RuntimeError('Full training requires a GPU; CPU is supported for smoke checks only')
    if device.type=='cuda' and not torch.cuda.is_available():raise RuntimeError('Enable a GPU runtime first')
    model=OCTCNN(args.variant).to(device)
    if args.variant not in ('basic','basic_noise','dynamic','na_dynamic'):raise ValueError('Variant outside predeclared factorial batch')
    train_data=OCTDataset(cfg['audit_dir'],'OCT2017_train',True,args.seed,normalize=False)
    val_data=OCTDataset(cfg['audit_dir'],'OCT2017_val',False,args.seed)
    if args.smoke:
        for dataset in [train_data,val_data]:
            indices=np.concatenate([np.flatnonzero(dataset.labels==c)[:64] for c in range(4)])
            dataset.rows=[dataset.rows[i] for i in indices];dataset.labels=dataset.labels[indices]
    val_loader=DataLoader(val_data,batch_size=cfg['batch_size'],shuffle=False,num_workers=cfg['workers'],pin_memory=True)
    counts=np.bincount(train_data.labels,minlength=4)
    if np.any(counts==0):raise RuntimeError('Training requires all four classes')
    weights=len(train_data)/(4*counts.astype(float));weights=weights/weights.mean()
    criterion=torch.nn.CrossEntropyLoss(weight=torch.tensor(weights,dtype=torch.float32,device=device),label_smoothing=cfg['label_smoothing'])
    opt=torch.optim.AdamW(model.parameters(),lr=cfg['peak_lr'],weight_decay=cfg['weight_decay'])
    start=0;best=-1;bad_epochs=0;best_epoch=-1
    if args.resume and (out/'last.pt').exists():
        state=torch.load(out/'last.pt',map_location='cpu',weights_only=False)
        if state['config_hash']!=cfg_hash:raise RuntimeError('Resume configuration differs')
        previous=json.loads((out/'provenance.json').read_text())
        for key in ('OCT2017_train','OCT2017_val'):
            if previous['manifest_sha256'][key]!=audit['manifest_sha256'][key]:raise RuntimeError('Resume data membership changed')
        current_code={f.name:sha256_file(f) for f in Path(__file__).parent.glob('*.py')}
        if previous['code_sha256']!=current_code:raise RuntimeError('Resume code changed; restore the original commit or start a new run')
        model.load_state_dict(state['model']);opt.load_state_dict(state['optimizer'])
        # Move optimizer tensors after loading a CPU checkpoint.
        for values in opt.state.values():
            for k,v in values.items():
                if torch.is_tensor(v):values[k]=v.to(device)
        start=state['epoch']+1;best=state['best'];bad_epochs=state['bad_epochs'];best_epoch=state['best_epoch']
        torch.set_rng_state(state['torch_rng']);torch.cuda.set_rng_state_all(state['cuda_rng']) if device.type=='cuda' else None
        random.setstate(state['python_rng']);np.random.set_state(state['numpy_rng'])
    write_json(out/'config.json',cfg)
    metadata={'config_hash':cfg_hash,'audit_sha256':sha256_file(Path(cfg['audit_dir'])/'audit.json'),
              'manifest_sha256':audit['manifest_sha256'],'python':platform.python_version(),'torch':torch.__version__,
              'numpy':np.__version__,'cuda_runtime':torch.version.cuda,'gpu':torch.cuda.get_device_name() if device.type=='cuda' else 'CPU smoke',
              'slurm_job_id':os.environ.get('SLURM_JOB_ID'),'hostname':platform.node(),
              'driver':subprocess.check_output(['nvidia-smi','--query-gpu=driver_version','--format=csv,noheader'],text=True).strip() if device.type=='cuda' else None,
              'class_weights':weights.tolist(),'class_order':CLASSES,'complexity':complexity(args.variant),
              'code_sha256':{f.name:sha256_file(f) for f in Path(__file__).parent.glob('*.py')}}
    write_json(out/'provenance.json',metadata)
    print(json.dumps(metadata),flush=True)
    max_epochs=2 if args.smoke else cfg['max_epochs'];batch_size=cfg['batch_size']
    for epoch in range(start,max_epochs):
        if epoch<cfg['warmup_epochs']:
            lr=cfg['min_lr']+(cfg['peak_lr']-cfg['min_lr'])*epoch/max(1,cfg['warmup_epochs']-1)
        else:
            phase=(epoch-cfg['warmup_epochs'])/max(1,cfg['max_epochs']-cfg['warmup_epochs']-1)
            lr=cfg['min_lr']+(cfg['peak_lr']-cfg['min_lr'])*(1+math.cos(math.pi*phase))/2
        for group in opt.param_groups:group['lr']=lr
        train_data.epoch=epoch;model.set_epoch(epoch);model.train()
        generator=torch.Generator().manual_seed(args.seed*1000003+epoch)
        loader=DataLoader(train_data,batch_size=batch_size,shuffle=True,generator=generator,num_workers=cfg['workers'],pin_memory=True)
        elapsed=time.monotonic();loss_sum=0;noise_sum=0;total_sum=0;seen=0
        torch.cuda.reset_peak_memory_stats() if device.type=='cuda' else None
        for step,(x,y,_) in enumerate(loader):
            x=x.to(device,non_blocking=True);y=y.to(device,non_blocking=True);opt.zero_grad(set_to_none=True)
            noise_generator=make_noise_generator(device,args.seed,epoch,step)
            losses=backward_objective(model,x,y,criterion,cfg,args.variant,noise_generator)
            torch.nn.utils.clip_grad_norm_(model.parameters(),cfg['gradient_clip'],error_if_nonfinite=True);opt.step()
            loss_sum+=losses['clean']*len(y);noise_sum+=losses['noise']*len(y);total_sum+=losses['total']*len(y);seen+=len(y)
            if step and step%200==0:print(json.dumps({'epoch':epoch+1,'step':step,'of':len(loader),'train_loss':loss_sum/seen}),flush=True)
        torch.cuda.synchronize() if device.type=='cuda' else None
        train_seconds=time.monotonic()-elapsed
        metrics=validate(model,val_loader,device);score=metrics['balanced_accuracy']
        improved=score>best+cfg['min_delta']
        if improved:best=score;best_epoch=epoch;bad_epochs=0
        else:bad_epochs+=1
        checkpoint={'model':model.state_dict(),'optimizer':opt.state_dict(),'epoch':epoch,'best':best,'best_epoch':best_epoch,
                    'bad_epochs':bad_epochs,'config':cfg,'config_hash':cfg_hash,'validation':metrics,
                    'torch_rng':torch.get_rng_state(),'cuda_rng':torch.cuda.get_rng_state_all() if device.type=='cuda' else [],
                    'python_rng':random.getstate(),'numpy_rng':np.random.get_state()}
        atomic_save(checkpoint,out/'last.pt')
        if improved:atomic_save(checkpoint,out/'best.pt')
        record={'epoch':epoch+1,'lr':lr,'train_loss':loss_sum/seen,'noise_loss':noise_sum/seen,'total_loss':total_sum/seen,
                'training_seconds':train_seconds,'peak_cuda_memory_bytes':torch.cuda.max_memory_allocated() if device.type=='cuda' else 0,
                'validation':metrics,'best_epoch':best_epoch+1,'bad_epochs':bad_epochs}
        with (out/'history.jsonl').open('a') as f:f.write(json.dumps(record)+'\n')
        print(json.dumps(record),flush=True);sync_small(out,args.archive)
        if bad_epochs>=cfg['patience']:break
    state=torch.load(out/'best.pt',map_location=device,weights_only=False)
    model.load_state_dict(state['model']);model.set_epoch(state['epoch'])
    reproduced=validate(model,val_loader,device,out/'validation_predictions.csv')
    if reproduced['confusion_matrix']!=state['validation']['confusion_matrix']:raise RuntimeError('Best checkpoint reload validation mismatch')
    write_json(out/'completed.json',{'best_epoch':best_epoch+1,'best_validation':reproduced,'epochs_completed':epoch+1,
                                   'best_checkpoint_sha256':sha256_file(out/'best.pt'),'config_hash':cfg_hash,'smoke':args.smoke})
    sync_small(out,args.archive)

if __name__=='__main__':main()
