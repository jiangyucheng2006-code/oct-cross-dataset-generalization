"""Shared, source-controlled OCT preprocessing and evaluation contracts."""
import hashlib
import json
from pathlib import Path
import numpy as np
from PIL import Image, ImageOps

CLASSES = ('CNV', 'DME', 'DRUSEN', 'NORMAL')
NEH_INDICES = (0, 2, 3)
IMAGE_SIZE = 256

def sha256_file(path):
    h = hashlib.sha256()
    with open(path, 'rb') as f:
        for chunk in iter(lambda: f.read(4 * 1024 * 1024), b''):
            h.update(chunk)
    return h.hexdigest()

def write_json(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + '.tmp')
    temporary.write_text(json.dumps(value, indent=2, ensure_ascii=False, allow_nan=False), encoding='utf8')
    temporary.replace(path)

def grayscale8(image):
    # The paper specifies linear scaling of 16-bit exports but not its endpoints.
    # Use the full 16-bit range, not per-image min/max. Record this assumption.
    if image.mode.startswith('I;16') or image.mode == 'I':
        a = np.asarray(image, dtype=np.float32)
        if a.min() < 0 or a.max() > 65535:
            raise ValueError('Integer image is outside the declared 16-bit range')
        return Image.fromarray(np.rint(a / 65535 * 255).astype('uint8'))
    if image.mode == 'F':
        raise ValueError('Float input needs an explicit intensity scaling rule')
    return image.convert('L')

def square_resize(image, size=IMAGE_SIZE):
    image = grayscale8(image)
    w, h = image.size
    d = max(w, h)
    image = ImageOps.expand(image, ((d-w)//2, (d-h)//2, d-w-(d-w)//2, d-h-(d-h)//2), fill=0)
    return image.resize((size, size), Image.Resampling.LANCZOS)

def metrics_from_probs(labels, probabilities, indices=range(4)):
    """Metrics use global labels and an explicitly declared output class set."""
    indices = tuple(indices)
    lookup = {k: i for i, k in enumerate(indices)}
    y = np.asarray([lookup[int(v)] for v in labels], dtype=np.int64)
    p = np.asarray(probabilities, dtype=np.float64)
    assert p.shape == (len(y), len(indices))
    assert np.isfinite(p).all() and np.allclose(p.sum(1), 1, atol=1e-5)
    pred = p.argmax(1)
    cm = np.bincount(y * len(indices) + pred, minlength=len(indices)**2).reshape(len(indices), len(indices))
    tp = cm.diagonal().astype(float); support = cm.sum(1); proposed = cm.sum(0)
    recall = np.divide(tp, support, out=np.zeros_like(tp), where=support>0)
    precision = np.divide(tp, proposed, out=np.zeros_like(tp), where=proposed>0)
    f1 = np.divide(2*tp, support+proposed, out=np.zeros_like(tp), where=(support+proposed)>0)
    f2 = np.divide(5*tp, 4*support+proposed, out=np.zeros_like(tp), where=(4*support+proposed)>0)
    tn = len(y) - support - proposed + tp
    specificity = np.divide(tn, len(y)-support, out=np.zeros_like(tp), where=(len(y)-support)>0)
    acc = float(tp.sum()/len(y)); expected = float((support*proposed).sum()/len(y)**2)
    order = np.argsort(p.max(1), kind='stable')
    ece = sum(len(b)/len(y)*abs(float((pred[b]==y[b]).mean())-float(p[b].max(1).mean())) for b in np.array_split(order, min(15,len(y))) if len(b))
    entropy = -(p*np.log(p.clip(1e-12))).sum(1)
    confident = np.argsort(entropy, kind='stable')[:max(1,int(len(y)*.8))]
    return {'n': len(y), 'accuracy': acc, 'balanced_accuracy': float(recall.mean()),
            'macro_f1': float(f1.mean()), 'weighted_f2': float((f2*support).sum()/len(y)),
            'macro_specificity': float(specificity.mean()), 'kappa': (acc-expected)/(1-expected) if expected<1 else None,
            'ece_15_equal_mass': float(ece), 'risk_at_80pct_coverage': float((pred[confident]!=y[confident]).mean()),
            'classes': [CLASSES[k] for k in indices], 'support': support.tolist(), 'recall': recall.tolist(),
            'precision': precision.tolist(), 'per_class_f1': f1.tolist(), 'confusion_matrix': cm.tolist()}
