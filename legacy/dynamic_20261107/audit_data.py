"""Read-only audit of master images; create split manifests and disposable caches."""
import argparse, csv, hashlib, io, json, re, struct
from collections import Counter, defaultdict
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path
import numpy as np
from PIL import Image
from common import CLASSES, IMAGE_SIZE, square_resize, write_json, sha256_file

SUFFIXES = {'.jpg','.jpeg','.png','.tif','.tiff','.bmp'}
EXPECTED = {'OCT2017_train':60663, 'OCT2017_val':20246, 'OCT2017_test':20240,
            'NEH_external_test':16663, 'OCT-C8_external_test':1546}

def inspect_image(row):
    row = dict(row)
    try:
        raw = Path(row['path']).read_bytes()
        row['raw_sha256'] = hashlib.sha256(raw).hexdigest()
        with Image.open(io.BytesIO(raw)) as im:
            im.load(); row['width'], row['height'] = im.size; row['mode'] = im.mode
            # Content hashing uses original decoded pixels before resizing/scaling.
            a = np.asarray(im)
            content = str((a.shape,a.dtype.str)).encode() + a.tobytes()
            row['pixel_sha256'] = hashlib.sha256(content).hexdigest()
            gray = np.asarray(im.convert('L'))
            row['gray_sha256'] = hashlib.sha256(struct.pack('>II',im.width,im.height)+gray.tobytes()).hexdigest()
            resized = np.asarray(square_resize(im), dtype=np.uint8)
        return row, resized, None
    except Exception as e:
        return row, None, repr(e)

def main():
    parser=argparse.ArgumentParser();parser.add_argument('--root',required=True);parser.add_argument('--out',required=True);parser.add_argument('--workers',type=int,default=4)
    args=parser.parse_args();root=Path(args.root);out=Path(args.out);out.mkdir(parents=True,exist_ok=True)
    records=[];excluded=Counter();patients=defaultdict(set);errors=[];counts={}; cache_paths={}
    for dataset,splits in [('OCT2017',['train','val','test']),('NEH',['external_test']),('OCT-C8',['external_test'])]:
        for split in splits:
            key=dataset+'_'+split;rows=[]
            for path in sorted((root/dataset/split).rglob('*')):
                if not path.is_file() or path.suffix.lower() not in SUFFIXES:continue
                label=path.relative_to(root/dataset/split).parts[0].upper()
                if label not in CLASSES:
                    excluded[dataset+'/'+label]+=1;continue
                pid=''
                if dataset=='OCT2017':
                    match=re.search(r'(?:CNV|DME|DRUSEN|NORMAL)-(\d+)-',path.name,re.I)
                    if not match:errors.append({'path':str(path),'error':'unrecognized source patient ID'})
                    else:pid=match[1];patients[split].add(pid)
                elif dataset=='NEH':
                    match=re.match(r'([A-Za-z]+)__(\d+)__',path.name)
                    if match:pid=match[1].upper()+'_'+match[2]
                rows.append({'dataset':dataset,'split':split,'label':label,'class_index':CLASSES.index(label),'path':str(path),
                             'patient_id':pid,'cache_index':len(rows),'cache_key':key})
            counts[key]=len(rows)
            if len(rows)!=EXPECTED[key]:errors.append({'key':key,'expected':EXPECTED[key],'found':len(rows)})
            cache=out/(key+'.npy');cache_paths[key]=str(cache)
            mm=np.lib.format.open_memmap(cache,mode='w+',dtype='uint8',shape=(len(rows),IMAGE_SIZE,IMAGE_SIZE))
            with ProcessPoolExecutor(max_workers=args.workers) as pool:
                for start in range(0,len(rows),1024):
                    for row,array,error in pool.map(inspect_image,rows[start:start+1024],chunksize=16):
                        if error:errors.append({'path':row['path'],'error':error})
                        else:mm[row['cache_index']]=array;records.append(row)
                    if start%10240==0:print(key,min(start+1024,len(rows)),'/',len(rows),flush=True)
            mm.flush();del mm
    overlap={a+'__'+b:sorted(patients[a]&patients[b]) for a,b in [('train','val'),('train','test'),('val','test')]}
    source_cross=[];exclusions={}; duplicates=[]
    for hashkey in ['raw_sha256','pixel_sha256','gray_sha256']:
        groups=defaultdict(list)
        for r in records:groups[r[hashkey]].append(r)
        for digest,group in groups.items():
            if len(group)<2:continue
            source=[r for r in group if r['dataset']=='OCT2017']
            external=[r for r in group if r['dataset']!='OCT2017']
            if len({r['split'] for r in source})>1:
                source_cross.append({'hash_type':hashkey,'hash':digest,'paths':[r['path'] for r in source]})
            if len({r['label'] for r in source})>1:
                errors.append({'error':'source label conflict','paths':[r['path'] for r in source]})
            if source:
                for r in external:exclusions[r['path']]='exact content shared with OCT2017'
            else:
                # Exact duplicates shared between the external datasets also reduce independence.
                # Keep NEH in preference to OCT-C8 (predeclared before predictions).
                group=sorted(group,key=lambda r:(r['dataset']!='NEH',r['path']))
                if len({r['label'] for r in group})>1:
                    for r in group:exclusions[r['path']]='conflicting labels on identical external content'
                else:
                    for r in group[1:]:exclusions[r['path']]='duplicate external content; keep NEH then lexicographic first'
            duplicates.append({'hash_type':hashkey,'hash':digest,'paths':[r['path'] for r in group]})
    fields=list(records[0])
    all_path=out/'all_records.csv'
    with all_path.open('w',newline='') as f:
        w=csv.DictWriter(f,fieldnames=fields);w.writeheader();w.writerows(records)
    manifest_hashes={};retained_counts={}
    for key in counts:
        selected=[r for r in records if r['cache_key']==key and r['path'] not in exclusions]
        p=out/(key+'.csv')
        with p.open('w',newline='') as f:
            w=csv.DictWriter(f,fieldnames=fields);w.writeheader();w.writerows(selected)
        manifest_hashes[key]=sha256_file(p);retained_counts[key]=dict(Counter(r['label'] for r in selected))
    write_json(out/'exact_duplicate_groups.json',duplicates)
    write_json(out/'external_exclusions.json',exclusions)
    passed=not errors and not source_cross and not any(overlap.values())
    report={'passed':passed,'input_counts':counts,'retained_counts':retained_counts,'excluded_nonshared_classes':dict(excluded),
            'source_patient_counts':{k:len(v) for k,v in patients.items()},'source_patient_overlaps':overlap,
            'source_cross_split_exact_matches':source_cross,'errors':errors,'external_exclusion_count':len(exclusions),
            'manifest_sha256':manifest_hashes,'cache_paths':cache_paths,'image_size':IMAGE_SIZE,
            'limitations':['Patient IDs are filename-derived, not clinically verified.','Near duplicates are not assessed.',
                            'OCT-C8 patient IDs are unavailable; do not claim patient-level external independence.',
                            'NEH patient grouping uses original cohort prefix plus filename numeric patient ID.']}
    write_json(out/'audit.json',report)
    print(json.dumps({k:v for k,v in report.items() if k not in ['cache_paths','manifest_sha256']},indent=2),flush=True)
    if not passed:raise SystemExit(2)

if __name__=='__main__':main()
