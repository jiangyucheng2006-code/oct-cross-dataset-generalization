"""Freeze source-selected checkpoints, then score the three predeclared test sets."""
import argparse, csv, json, math, platform, time
from pathlib import Path
import numpy as np
import torch
from torch.utils.data import DataLoader
from common import CLASSES, NEH_INDICES, write_json, sha256_file, metrics_from_probs
from data import OCTDataset
from models import OCTCNN
from train import seed_everything, sync_small

def bootstrap_intervals(labels,probabilities,rows,indices,dataset,repeats=1000):
    """Within-run uncertainty; seed variation is reported separately."""
    mapping={k:i for i,k in enumerate(indices)};y=np.asarray([mapping[int(v)] for v in labels]);pred=probabilities.argmax(1)
    k=len(indices);rng=np.random.default_rng(20261107);acc=[];mf1=[]
    if dataset=='OCT-C8':
        # C8 has no reliable patient IDs: image-stratified interval, explicitly limited.
        groups=[np.flatnonzero(y==c) for c in range(k)]
        for _ in range(repeats):
            chosen=np.concatenate([rng.choice(g,len(g),replace=True) for g in groups if len(g)])
            cm=np.bincount(y[chosen]*k+pred[chosen],minlength=k*k).reshape(k,k)
            tp=cm.diagonal();den=cm.sum(0)+cm.sum(1)
            acc.append(float(tp.sum()/cm.sum()));mf1.append(float(np.divide(2*tp,den,out=np.zeros(k),where=den>0).mean()))
        unit='class-stratified image bootstrap; patient independence unverified'
    else:
        ids=[r['patient_id'] for r in rows]
        if any(not v for v in ids):return {'available':False,'reason':'patient ID parsing incomplete'}
        _,group=np.unique(ids,return_inverse=True);n=int(group.max())+1
        cms=np.bincount(group*k*k+y*k+pred,minlength=n*k*k).reshape(n,k,k)
        for _ in range(repeats):
            cm=cms[rng.integers(0,n,size=n)].sum(0);tp=cm.diagonal();den=cm.sum(0)+cm.sum(1)
            acc.append(float(tp.sum()/cm.sum()));mf1.append(float(np.divide(2*tp,den,out=np.zeros(k),where=den>0).mean()))
        unit='filename-derived patient cluster bootstrap'
    return {'available':True,'unit':unit,'repeats':repeats,'confidence_level':.95,
            'accuracy':np.quantile(acc,[.025,.975]).tolist(),'macro_f1':np.quantile(mf1,[.025,.975]).tolist(),
            'scope':'within-run sampling uncertainty; does not include between-seed variation or annotation uncertainty'}

def freeze_runs(root,experiments):
    entries=[]
    for variant in experiments['variants']:
        for seed in experiments['seeds']:
            run=root/f'{variant}_seed{seed}'
            done=json.loads((run/'completed.json').read_text())
            if done['smoke']:raise RuntimeError('Smoke checkpoint cannot enter formal evaluation')
            digest=sha256_file(run/'best.pt')
            if digest!=done['best_checkpoint_sha256']:raise RuntimeError('Checkpoint changed after completion')
            entries.append({'variant':variant,'seed':seed,'run':str(run),'checkpoint_sha256':digest,
                            'validation_balanced_accuracy':done['best_validation']['balanced_accuracy'],
                            'best_epoch':done['best_epoch']})
    means={v:float(np.mean([r['validation_balanced_accuracy'] for r in entries if r['variant']==v])) for v in experiments['variants']}
    frozen={'runs':entries,'source_validation_means':means,'selected_by_source_validation':max(means,key=means.get),
            'external_tests':['OCT2017_test','NEH_external_test','OCT-C8_external_test'],
            'selection_rule':'mean OCT2017 validation balanced accuracy; tie follows predeclared order'}
    p=root/'frozen_before_test.json'
    if p.exists():
        if json.loads(p.read_text())!=frozen:raise RuntimeError('Previously frozen checkpoint list differs')
    else:write_json(p,frozen)
    return frozen

@torch.inference_mode()
def score_run(entry,cfg,archive):
    run=Path(entry['run']);destination=run/'evaluation'
    if (destination/'complete.json').exists():return json.loads((destination/'complete.json').read_text())
    destination.mkdir(parents=True,exist_ok=True);seed_everything(entry['seed'])
    ckpt=torch.load(run/'best.pt',map_location='cuda',weights_only=False)
    model=OCTCNN(entry['variant']).cuda();model.load_state_dict(ckpt['model']);model.set_epoch(ckpt['epoch']);model.eval()
    result={'variant':entry['variant'],'seed':entry['seed'],'checkpoint_sha256':entry['checkpoint_sha256'],
            'gpu':torch.cuda.get_device_name(),'hostname':platform.node(),'datasets':{}}
    audit=json.loads((Path(cfg['audit_dir'])/'audit.json').read_text())
    for key in ['OCT2017_test','NEH_external_test','OCT-C8_external_test']:
        if sha256_file(Path(cfg['audit_dir'])/(key+'.csv'))!=audit['manifest_sha256'][key]:raise RuntimeError('Test manifest changed')
        data=OCTDataset(cfg['audit_dir'],key);loader=DataLoader(data,batch_size=cfg['batch_size'],num_workers=cfg['workers'],pin_memory=True,shuffle=False)
        selected=NEH_INDICES if key.startswith('NEH') else tuple(range(4))
        all_raw=[];all_restricted=[];labels=[];sample_indices=[]
        for x,y,row_index in loader:
            logits=model(x.cuda(non_blocking=True))
            all_raw.append(logits.softmax(1).cpu().numpy())
            all_restricted.append(logits[:,selected].softmax(1).cpu().numpy())
            labels.append(y.numpy());sample_indices.extend(row_index.tolist())
        raw=np.concatenate(all_raw);probs=np.concatenate(all_restricted);labels=np.concatenate(labels)
        rows=[data.rows[i] for i in sample_indices]
        metrics=metrics_from_probs(labels,probs,selected)
        dataset_name=key.split('_')[0]
        metrics['bootstrap_95ci']=bootstrap_intervals(labels,probs,rows,selected,dataset_name)
        metrics['evaluation_rule']='known-target-label-set: drop DME then renormalize' if key.startswith('NEH') else 'unrestricted four-class argmax'
        if key.startswith('NEH'):
            metrics['raw_four_class_accuracy_diagnostic_only']=float((raw.argmax(1)==labels).mean())
            metrics['raw_DME_prediction_fraction']=float((raw.argmax(1)==1).mean())
        if key.startswith('OCT-C8'):
            metrics['majority_class_accuracy']=float(max(metrics['support'])/len(labels))
        fields=['filename','path','patient_id','true_class','prediction','raw_four_class_prediction']+['raw_p_'+c for c in CLASSES]+['eval_p_'+CLASSES[c] for c in selected]
        with (destination/(key+'_predictions.csv')).open('w',newline='') as f:
            writer=csv.DictWriter(f,fieldnames=fields);writer.writeheader()
            for i,row in enumerate(rows):
                r={'filename':Path(row['path']).name,'path':row['path'],'patient_id':row['patient_id'],
                   'true_class':CLASSES[int(labels[i])],'prediction':CLASSES[selected[int(probs[i].argmax())]],
                   'raw_four_class_prediction':CLASSES[int(raw[i].argmax())]}
                r.update({'raw_p_'+c:float(raw[i,j]) for j,c in enumerate(CLASSES)})
                r.update({'eval_p_'+CLASSES[c]:float(probs[i,j]) for j,c in enumerate(selected)})
                writer.writerow(r)
        result['datasets'][key]=metrics
        print(json.dumps({'variant':entry['variant'],'seed':entry['seed'],'dataset':key,'metrics':metrics}),flush=True)
    # Comparable forward-only timing, on the same GPU and precision for all runs.
    sample=torch.stack([data[i][0] for i in range(min(64,len(data)))]).cuda()
    for _ in range(10):model(sample)
    timings=[]
    for _ in range(50):
        start=torch.cuda.Event(enable_timing=True);end=torch.cuda.Event(enable_timing=True)
        start.record();model(sample);end.record();torch.cuda.synchronize();timings.append(start.elapsed_time(end)/len(sample))
    result['timing']={'batch_size':len(sample),'precision':'FP32','forward_only_ms_per_image_median':float(np.median(timings)),
                      'iqr_ms':np.quantile(timings,[.25,.75]).tolist(),'rounds':50,'warmup':10,'not_single_image_latency':True}
    write_json(destination/'complete.json',result)
    sync_small(destination,Path(archive)/run.name)
    return result

def main():
    p=argparse.ArgumentParser();p.add_argument('--root',required=True);p.add_argument('--config',required=True);p.add_argument('--experiments',required=True);p.add_argument('--archive',required=True)
    args=p.parse_args();torch.set_num_threads(2);root=Path(args.root)
    cfg=json.loads(Path(args.config).read_text());exp=json.loads(Path(args.experiments).read_text())
    frozen=freeze_runs(root,exp);sync_small(root,Path(args.archive).parent)
    results=[score_run(entry,cfg,args.archive) for entry in frozen['runs']]
    summary={}
    for variant in exp['variants']:
        summary[variant]={}
        for key in frozen['external_tests']:
            summary[variant][key]={}
            for metric in ['accuracy','balanced_accuracy','macro_f1','weighted_f2','kappa','ece_15_equal_mass','risk_at_80pct_coverage']:
                values=[r['datasets'][key][metric] for r in results if r['variant']==variant]
                summary[variant][key][metric]={'mean':float(np.mean(values)),'sample_sd':float(np.std(values,ddof=1)),'seeds':exp['seeds'],'values':values}
    write_json(root/'summary.json',{'results':summary,'source_selected_variant':frozen['selected_by_source_validation'],
                                   'all_predeclared_variants_reported':True,'no_external_selection':True})
    sync_small(root,Path(args.archive).parent)

if __name__=='__main__':main()
