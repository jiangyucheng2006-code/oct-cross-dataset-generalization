"""Read cleaned ZIPs in place, preserve their split, and build disposable caches.

No image is extracted, moved or deleted. Full mode checks the team's archive
SHA256 values. Smoke mode reads a deterministic 64/class source subset only.
"""
import argparse
from collections import Counter, defaultdict
import csv
import hashlib
import io
import json
from pathlib import Path, PurePosixPath
import re
import struct
import zipfile

import numpy as np
from PIL import Image
from common import CLASSES, IMAGE_SIZE, sha256_file, square_resize, write_json

SUFFIXES = {'.jpg', '.jpeg', '.png', '.bmp', '.tif', '.tiff'}
FIELDS = ['dataset', 'split', 'label', 'class_index', 'path', 'patient_id',
          'cache_index', 'cache_key', 'width', 'height', 'mode',
          'raw_sha256', 'pixel_sha256', 'gray_sha256']


def index_archive(path, dataset):
    """Handle wrapper folders, but never infer or generate a new source split."""
    rows = []
    with zipfile.ZipFile(path) as archive:
        if len(archive.namelist()) != len(set(archive.namelist())):
            raise ValueError('Duplicate ZIP member names: ' + str(path))
        for info in sorted(archive.infolist(), key=lambda item: item.filename):
            name = info.filename.replace('\\', '/')
            p = PurePosixPath(name)
            if info.is_dir() or p.suffix.lower() not in SUFFIXES or '__MACOSX' in p.parts:
                continue
            if p.is_absolute() or '..' in p.parts:
                raise ValueError('Unsafe archive member name: ' + name)
            # Class folder directly contains the image, independent of wrappers.
            label = p.parent.name.upper()
            if label not in CLASSES:
                if dataset == 'OCT2017':
                    raise ValueError('Unknown source class directory: ' + name)
                continue
            if dataset == 'OCT2017':
                splits = [part.lower() for part in p.parts[:-2] if part.lower() in ('train', 'val', 'test')]
                if len(splits) != 1:
                    raise ValueError('Expected one explicit train/val/test directory: ' + name)
                split = splits[0]
                match = re.search(r'(?:CNV|DME|DRUSEN|NORMAL)-(\d+)-', p.name, re.I)
                if not match:
                    raise ValueError('Unrecognized source patient ID: ' + name)
                patient = match[1]
            else:
                split = 'external_test'
                match = re.match(r'([A-Za-z]+)__(\d+)__', p.name) if dataset == 'NEH' else None
                patient = match[1].upper() + '_' + match[2] if match else ''
                if dataset == 'NEH' and label == 'DME':
                    raise ValueError('NEH unexpectedly contains DME')
            rows.append({'dataset': dataset, 'split': split, 'label': label,
                         'class_index': CLASSES.index(label), 'path': dataset + '/' + name,
                         'patient_id': patient, 'member': info.filename,
                         'cache_key': dataset + '_' + split})
    if not rows:
        raise ValueError('No supported images; use the inner cleaned ZIP, not a Drive folder ZIP: ' + str(path))
    return rows


def source_split_check(rows):
    patients = {split: set() for split in ('train', 'val', 'test')}
    for row in rows:
        if row['dataset'] == 'OCT2017':
            patients[row['split']].add(row['patient_id'])
    overlaps = {a + '__' + b: sorted(patients[a] & patients[b])
                for a, b in [('train', 'val'), ('train', 'test'), ('val', 'test')]}
    if any(overlaps.values()):
        raise ValueError('Source patient IDs overlap across splits; do not train: ' + str(overlaps))
    return patients, overlaps


def inspect_bytes(raw, row):
    result = {k: v for k, v in row.items() if k != 'member'}
    result['raw_sha256'] = hashlib.sha256(raw).hexdigest()
    with Image.open(io.BytesIO(raw)) as im:
        im.load()
        result['width'], result['height'] = im.size
        result['mode'] = im.mode
        a = np.asarray(im)
        result['pixel_sha256'] = hashlib.sha256(str((a.shape, a.dtype.str)).encode() + a.tobytes()).hexdigest()
        gray = np.asarray(im.convert('L'))
        result['gray_sha256'] = hashlib.sha256(struct.pack('>II', im.width, im.height) + gray.tobytes()).hexdigest()
        array = np.asarray(square_resize(im), dtype=np.uint8)
    return result, array


def duplicate_rules(records):
    """Hash connected components, including transitive cross-label conflicts."""
    parent = list(range(len(records)))

    def find(a):
        while parent[a] != a:
            parent[a] = parent[parent[a]]
            a = parent[a]
        return a

    for hash_key in ('raw_sha256', 'pixel_sha256', 'gray_sha256'):
        first = {}
        for i, row in enumerate(records):
            digest = row[hash_key]
            if digest in first:
                a, b = find(i), find(first[digest])
                parent[a] = b
            else:
                first[digest] = i
    groups = defaultdict(list)
    for i, row in enumerate(records):
        groups[find(i)].append(row)
    excluded, errors, duplicates = {}, [], []
    for group in groups.values():
        if len(group) < 2:
            continue
        source = [r for r in group if r['dataset'] == 'OCT2017']
        external = [r for r in group if r['dataset'] != 'OCT2017']
        if len({r['split'] for r in source}) > 1:
            errors.append('Exact source content crosses splits: ' + str([r['path'] for r in source]))
        if len({r['label'] for r in source}) > 1:
            errors.append('Conflicting source labels: ' + str([r['path'] for r in source]))
        if source:
            for row in external:
                excluded[row['path']] = 'Exact content shared with source'
        elif len({r['label'] for r in external}) > 1:
            for row in external:
                excluded[row['path']] = 'Conflicting external labels'
        else:
            ordered = sorted(external, key=lambda r: (r['dataset'] != 'NEH', r['path']))
            for row in ordered[1:]:
                excluded[row['path']] = 'Exact duplicate; keep NEH then lexicographic first'
        duplicates.append([r['path'] for r in group])
    return excluded, errors, duplicates


def prepare(archives, out, mode='smoke', specification=None, per_class=64):
    out = Path(out)
    out.mkdir(parents=True, exist_ok=True)
    if mode not in ('smoke', 'full'):
        raise ValueError(mode)
    if 'OCT2017' not in archives or (mode == 'full' and set(archives) != {'OCT2017', 'NEH', 'OCT-C8'}):
        raise ValueError('Full preparation needs all three archives; smoke needs OCT2017')
    selected_archives = {'OCT2017': Path(archives['OCT2017'])} if mode == 'smoke' else {k: Path(v) for k, v in archives.items()}
    indexed, archive_hashes, signatures = [], {}, {}
    for dataset, path in selected_archives.items():
        if not path.is_file():
            raise FileNotFoundError(path)
        signatures[dataset] = {'bytes': path.stat().st_size, 'mtime_ns': path.stat().st_mtime_ns}
        if mode == 'full':
            print('Checking archive SHA256:', dataset, flush=True)
            archive_hashes[dataset] = sha256_file(path)
            if specification and archive_hashes[dataset] != specification[dataset]['sha256']:
                raise ValueError(dataset + ' differs from the frozen team archive; obtain the correct cleaned ZIP')
        rows = index_archive(path, dataset)
        counts = dict(Counter(r['split'] for r in rows))
        if specification and counts != specification[dataset]['counts']:
            raise ValueError(f'{dataset} split counts differ: {counts}')
        indexed.extend(rows)
    patients, overlap = source_split_check(indexed)
    contract = {'mode': mode, 'per_class': per_class if mode == 'smoke' else None,
                'archive_signatures': signatures, 'archive_sha256': archive_hashes,
                'image_size': IMAGE_SIZE, 'preparation_code_sha256': sha256_file(__file__)}
    if (out / 'audit.json').exists():
        report = json.loads((out / 'audit.json').read_text())
        if not report['passed'] or report['preparation_contract'] != contract:
            raise ValueError('Existing cache differs or failed. Choose a NEW output folder; masters are untouched.')
        for key, digest in report['manifest_sha256'].items():
            if sha256_file(out / (key + '.csv')) != digest or not (out / (key + '.npy')).exists():
                raise ValueError('Cached manifest/data changed: ' + key)
        print('Reusing matching prepared cache:', out, flush=True)
        return report
    if any(out.iterdir()):
        raise ValueError('Incomplete/nonempty cache folder. Choose a new output folder.')
    original_counts = dict(Counter(r['cache_key'] for r in indexed))
    if mode == 'smoke':
        chosen = []
        for split in ('train', 'val'):
            for label in CLASSES:
                chosen += [r for r in indexed if r['split'] == split and r['label'] == label][:per_class]
        indexed = chosen
    records, counts = [], {}
    for dataset, path in selected_archives.items():
        with zipfile.ZipFile(path) as archive:
            for key in sorted({r['cache_key'] for r in indexed if r['dataset'] == dataset}):
                rows = [r for r in indexed if r['cache_key'] == key]
                counts[key] = len(rows)
                if not rows:
                    raise ValueError('Empty dataset: ' + key)
                cache = np.lib.format.open_memmap(out / (key + '.npy'), mode='w+', dtype='uint8', shape=(len(rows), IMAGE_SIZE, IMAGE_SIZE))
                for i, row in enumerate(rows):
                    row['cache_index'] = i
                    record, array = inspect_bytes(archive.read(row['member']), row)
                    cache[i] = array
                    records.append(record)
                    if (i + 1) % 5000 == 0:
                        print(key, i + 1, '/', len(rows), flush=True)
                cache.flush()
                del cache
                print('Prepared', key, len(rows), flush=True)
    exclusions, errors, duplicates = duplicate_rules(records)
    manifests, retained = {}, {}
    for key in counts:
        rows = [r for r in records if r['cache_key'] == key and r['path'] not in exclusions]
        with (out / (key + '.csv')).open('w', newline='') as handle:
            writer = csv.DictWriter(handle, fieldnames=FIELDS)
            writer.writeheader()
            writer.writerows(rows)
        manifests[key] = sha256_file(out / (key + '.csv'))
        retained[key] = dict(Counter(r['label'] for r in rows))
        expected_labels = set(CLASSES) - ({'DME'} if key.startswith('NEH') else set())
        if set(retained[key]) != expected_labels:
            errors.append('Missing shared class after exclusions: ' + key)
    report = {'passed': not errors, 'preparation_mode': mode, 'preparation_contract': contract,
              'input_counts': original_counts, 'prepared_counts': counts, 'retained_counts': retained,
              'source_patient_counts': {k: len(v) for k, v in patients.items()},
              'source_patient_overlaps': overlap, 'errors': errors, 'manifest_sha256': manifests,
              'external_exclusion_count': len(exclusions), 'image_size': IMAGE_SIZE,
              'limitations': ['Filename-derived IDs, not clinically verified.', 'Near duplicates are not checked.',
                              'C8 patient grouping unavailable.', 'Smoke audits exact content only in selected source samples.']}
    write_json(out / 'external_exclusions.json', exclusions)
    write_json(out / 'exact_duplicate_groups.json', duplicates)
    write_json(out / 'audit.json', report)
    if errors:
        raise ValueError('Data audit failed: ' + str(errors))
    return report


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--oct2017', required=True)
    parser.add_argument('--neh')
    parser.add_argument('--oct-c8')
    parser.add_argument('--out', required=True)
    parser.add_argument('--mode', choices=['smoke', 'full'], default='smoke')
    parser.add_argument('--spec', default=str(Path(__file__).resolve().parents[1] / 'configs/datasets.json'))
    args = parser.parse_args()
    paths = {'OCT2017': args.oct2017}
    if args.neh:
        paths['NEH'] = args.neh
    if args.oct_c8:
        paths['OCT-C8'] = args.oct_c8
    report = prepare(paths, args.out, args.mode, json.loads(Path(args.spec).read_text()))
    print(json.dumps({'passed': report['passed'], 'mode': report['preparation_mode'], 'counts': report['retained_counts']}, indent=2))


if __name__ == '__main__':
    main()
