# Method and reconstruction choices

## Main experiment

One source: OCT2017, 60,663 train / 20,246 validation / 20,240 test.
NEH and the four shared OCT-C8 classes are external tests. The new primary design
is a matched 2x2 comparison of static/dynamic convolution and clean/noise-aware
training. All groups use one fixed seed (42). No hyperparameter search has been
completed for this batch. Report every planned method and dataset.

Four feature blocks use channels 16, 32, 64, 128, followed by global average
pooling and a 128 -> 64 -> 4 classifier. A block contains 3x3 convolution,
BatchNorm, ReLU and 2x2 max pooling. Dynamic convolution mixes K=4 kernel experts
using an input-conditioned GAP/MLP softmax router at each layer. Parameter counts
are 105,908 (Basic) and 398,168 (Dynamic); noise training adds no inference
parameters. Input is one grayscale channel, black square padding, Lanczos resize
to 256x256, then per-image z-normalization.

The JBO article supports the four channel widths, expert count, GAP/classifier,
input size/padding and training schedule. Kernel size, pooling implementation,
BatchNorm, classifier hidden width, router bottleneck, augmentation strengths
and temperature schedule are explicit implementation choices where details
were not fully specified. The model is an independent reconstruction, not
official author code or an established exact reproduction.

## Noise-aware objective

For processed intensities x in [0,1], draw independent pixelwise noise
eta_n ~ N(0, 0.06^2), and use x_n = clip(x * (1 + eta_n), 0, 1).
Apply normalization independently to clean and noisy images **after** noise.

L = CE(clean) + (0.12 / 4) * sum[n=1..4] CE(noisy_n).

The constants sigma=0.06, beta=0.12 and N=4 follow the paper's OCT2017/D1
configuration, not its OCT-C8 configuration or illustration values. All views
share learned parameters and independently compute routing weights. One
optimizer step follows the clean and four noisy gradient contributions.
Sequential backward passes bound memory. BatchNorm receives clean then noisy
views sequentially; this ordering is a recorded choice. The perturbations model
exported intensity variability rather than a complete physical speckle process.
Validation/test use a single clean forward pass in eval mode.

## Common optimization

AdamW, batch 64, weight decay 1e-4; five-epoch linear warmup from 1e-6 to 1e-3,
cosine decay back to 1e-6, maximum 100 epochs, patience 10. Checkpoint selection
uses OCT2017 validation balanced accuracy only. Inverse-frequency class weights,
label smoothing 0.1 and gradient clipping at norm 5 are shared across four groups.
The paper mentions label smoothing without its coefficient: **0.1 is an
assumption**, not a quoted or optimized value. The earlier Dynamic-only batch
used smoothing 0.0, so it is retained as a separate historical protocol.

Common augmentation is horizontal flip p=0.5, rotation +/-5 degrees, scale
0.95-1.05, translation +/-3%, brightness/contrast factors 0.9-1.1. No vertical
flip or crop. Augmentation and noisy-view RNG streams are deterministic and
separate. FP32 and deterministic algorithms are used; different hardware and
software can still give different results. Normalization uses std.clamp_min(1e-6)
(the article writes std + epsilon). No target normalization is fitted.

## Evaluation

Freeze completed checkpoints before scoring all predeclared cohorts. NEH uses
known-target-label-set evaluation: drop DME logit and renormalize the remaining
three classes. Save raw four-class probabilities and accuracy as a diagnostic.
OCT-C8 uses all four output classes, restricted input labels only. Never choose
models from external scores. Metrics include accuracy, balanced accuracy,
macro-F1, support/per-class recall, weighted F2, specificity, kappa, equal-mass
ECE and risk at 80% coverage. No conformal guarantee is made.

Bootstrap intervals describe within-run sampling uncertainty: filename-derived
patient clusters for source/NEH where IDs parse; class-stratified image bootstrap
for C8 with patient independence unverified. One run gives no between-seed SD.
Forward-only GPU timing is amortized batch timing per image, not single-image
latency. Analytical MAC estimates list excluded operations; they are not the
paper's exact published FLOP count. Gains remain experimental outcomes.

## Exploratory router variants

`dynamic_stats` adds channel standard deviations to GAP router descriptors.
`dynamic_balance` adds 0.01 KL(batch-mean routing || uniform), averaged over
layers. These are source-only exploratory adaptations, not the paper's
noise-aware method. The first GAP router on a per-image z-normalized one-channel
input has little mean variation; deeper routers operate on learned maps.

See README references for the JBO paper and Chen et al. Dynamic Convolution.
