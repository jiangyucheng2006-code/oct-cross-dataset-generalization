"""Report the matched four-group experiment and historical runs separately.

No external result selects a model. One seed means no between-seed SD.
"""
import argparse
import csv
import json
from pathlib import Path
from common import write_json


METRICS=['accuracy','balanced_accuracy','macro_f1','weighted_f2','kappa','ece_15_equal_mass','risk_at_80pct_coverage']

def summarize(root,protocol):
    source=json.loads((root/'summary.json').read_text());rows=[]
    for variant,datasets in source['results'].items():
        for dataset,metrics in datasets.items():
            row={'protocol':protocol,'variant':variant,'dataset':dataset,
                 'n_runs':len(metrics['accuracy']['values']),
                 'seeds':','.join(str(s) for s in metrics['accuracy']['seeds'])}
            for metric in METRICS:
                row[metric]=metrics[metric]['mean'];row[metric+'_seed_sd']=metrics[metric]['sample_sd']
            rows.append(row)
    return rows,source

def main():
    p=argparse.ArgumentParser();p.add_argument('--root',required=True);p.add_argument('--historical')
    args=p.parse_args();ROOT=Path(args.root);HISTORICAL=Path(args.historical) if args.historical else None
    rows,current=summarize(ROOT,'matched_factorial_label_smoothing_0.1')
    historical_available=HISTORICAL is not None and (HISTORICAL/'summary.json').exists()
    if historical_available:
        old,_=summarize(HISTORICAL,'historical_fixed_recipe_label_smoothing_0.0');rows+=old
    with (ROOT/'combined_results.csv').open('w',newline='') as f:
        writer=csv.DictWriter(f,fieldnames=list(rows[0]));writer.writeheader();writer.writerows(rows)
    comparisons=[]
    for control,modified in [('basic','dynamic'),('basic','basic_noise'),('dynamic','na_dynamic'),('basic_noise','na_dynamic')]:
        if control not in current['results'] or modified not in current['results']:continue
        for dataset in current['results'][control]:
            deltas={metric:current['results'][modified][dataset][metric]['mean']-current['results'][control][dataset][metric]['mean'] for metric in METRICS}
            comparisons.append({'control':control,'modified':modified,'dataset':dataset,'difference_modified_minus_control':deltas})
    write_json(ROOT/'matched_contrasts.json',{'single_seed':True,'comparisons':comparisons,
        'interpretation':'Descriptive matched differences; not statistical evidence over training seeds.',
        'historical_batch_loaded':historical_available,'all_groups_reported':True})
    lines=['# OCT matched baseline and noise experiment','','One run per method (seed42). All test cohorts reported separately.',
           'Models/checkpoints selected only on OCT2017 validation data. No target adaptation.',
           'Primary four groups share label smoothing0.1. Historical variants used0.0 and are separate exploratory results.',
           'Bootstrap intervals in each evaluation/complete.json reflect within-run sampling uncertainty only.',
           '','| Protocol | Model | Dataset | Runs | Accuracy | Balanced accuracy | Macro-F1 |',
           '|---|---|---|---:|---:|---:|---:|']
    for row in rows:
        lines.append(f"| {row['protocol']} | {row['variant']} | {row['dataset']} | {row['n_runs']} | {row['accuracy']*100:.2f}% | {row['balanced_accuracy']*100:.2f}% | {row['macro_f1']*100:.2f}% |")
    lines+=['','NEH: DME logit removed and remaining three classes renormalized; raw four-class predictions also saved.',
            'OCT-C8: shared-class subset only; CNV10/DME7/DRUSEN86/NORMAL1443. Interpret macro metrics and per-class recall with sample-size uncertainty.',
            'No new dataset was added to the predeclared cohort list.']
    (ROOT/'RESULTS.md').write_text('\n'.join(lines)+'\n',encoding='utf8')
    print(json.dumps({'combined_result_rows':len(rows),'historical_available':historical_available,'single_seed_primary':True}))

if __name__=='__main__':main()
