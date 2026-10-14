# Architecture and data conventions

## Prediction task

For player p and pre-move context x, learn:

`P(move, duration | x, p) = P(move | x, p) P(duration | move, x, p)`.

Each training run now fixes p to one player and produces an independent checkpoint.
Hikaru and Caruana models start from the same pretrained Maia weights but have
separate trainable history/context layers, one player embedding and adapter,
move scorers, and timing heads. No trained weights or optimizer state are shared
between these runs. Older joint checkpoints can still be evaluated and loaded.

Inputs include board history, the last 16 plies with past durations, both clocks,
base time, increment, ratings and player identity. Current-move labels, final
results, engine evaluations and after-move clocks are never model inputs.

Maia's 64 square representations and legal candidate logits are extracted using
a temporary forward hook on its transformer. The upstream source revision is
pinned in `requirements-maia.txt`; weights load strictly with `weights_only=True`.
Its original ponder head is present in the frozen backbone but is not used as
our probability distribution. This is an adaptation layer, not a reproduction of
Maia's original population training.

## Trainable components

- Player embedding: 32 dimensions.
- History: square embeddings (16 each), promotion embedding (8), side indicator,
  log duration and missingness indicator, fed to a 64-dimensional GRU.
- Context fusion: pooled board representation + 15 numeric context features +
  player embedding + history, projected to 128 dimensions.
- Per-player adapter: 128 -> 32 -> 128 with a residual connection.
- Candidate scorer: source/destination square features + promotion + adapted
  context, projected to 128 dimensions.
- Policy correction: one scalar per legal move, initially zero.
- Timing head: 128 -> 64 -> 32 outputs per legal move.

Black moves are mirrored using python-chess square mirroring. Promotions are
represented explicitly, including underpromotions. Maia's own tokenization and
move vocabulary are used by its wrapper to preserve checkpoint compatibility.

## Timing labels

For single-stage Fischer increment games, `duration = before + increment - after`.
This estimates server-recorded clock consumption; it includes platform effects
and cannot recover unrecorded cognition, network latency or premove intent.

The first move by each side has no timing label. Missing clocks break the next
same-side difference: we never subtract a later clock from a stale clock. Negative
or over-budget durations are masked. Games without usable clocks can still train
the policy. Staged controls, correspondence games, unknown dates and nonstandard
variants are excluded by the current importer and counted in the audit.

Timing labels are binned as `duration / clock_before`. There are 32 bins: a
near-instant interval [0, 0.0001), followed by logarithmic intervals up to 1. This
handles different budgets without creating impossible over-clock predictions.
Reported means approximate each bin by its geometric center (the first by its
arithmetic midpoint). Within-bin timing and flagging behavior are not modeled.

## Training and evaluation

Train on observed moves with legal-move cross-entropy. Timing cross-entropy uses
only valid labels, conditioned on the observed move. Add `KL(base || adapted)`
regularization. Defaults: timing weight 0.5, KL weight 0.02, AdamW learning rate
0.0003, gradient clipping 1.0, three epochs. These are initial settings, not tuned
results. CUDA uses mixed precision; CPU tests use float32.

The dataset filters to the run's single target player before sampling; positions
from the other player do not train or validate that checkpoint. Training shuffles
without replacement using its own seeded random generator. All training positions
are used by default; optional caps use reservoir sampling across the source file.
Validation is never capped. A global chronological split keeps both sides of the same game
together, including games between the target players. Equal timestamps stay in
the same split. Validation
selects each model's best checkpoint using its player's move NLL plus weighted
marginal timing NLL. The shared trainer also retains macro-average selection for
older programmatic joint experiments. Test data is evaluated separately and never
used by training. The Colab workflow runs ten epochs per player; the library's
three-epoch default remains available for quick checks.

Metrics include move accuracy and NLL for both the adapted policy and frozen
backbone. Timing metrics include conditional NLL (knowing the actual move),
pre-move marginal NLL, approximate mean absolute error in seconds and coverage of
a nominal 80% interval expanded to bin boundaries. The latter can overcover due
to coarse bins. Results are grouped by player and time class.

Update-budget runs validate every `eval_every` successful optimizer updates and
at the final update, overriding epoch count. Separate move CE, conditional timing
CE, and KL training components are logged. A fixed seeded training subset uses
the same evaluator as validation; the raw training objective and marginal
validation score are not the same objective. The combined score still selects
`best.pt`; move NLL selects an additional `best-move.pt`.

Evaluation also groups by exact base+increment and remaining-clock band. It saves
game-identifiable validation records, approximate binned-median MAE, and interval
width alongside coverage. Empirical timing histograms and seconds medians are fit
only on selected training positions, with a player fallback for sparse or unseen
control/clock groups. Histogram smoothing gives nonzero probability to unseen
bins. The median-seconds baseline is bounded by the observed remaining clock.
Source hashes, selected offsets, batch-order hash and coverage are recorded.

## Important limits

The code has not been trained on full player archives. Synthetic test results are
software checks, not evidence of style fidelity or playing strength. No model
here has a measured Elo or established detection characteristics.

Caruana is a proposed second profile, not a claim that he moves more slowly than
Hikaru under matched conditions. A bullet-versus-rapid comparison measures time
control as well as player effects. Data from different years can also mix changes
in skill, repertoire, device and streaming conditions.

Maia pretraining may already contain older games used for downstream evaluation;
the chronological adaptation split cannot rule out upstream pretraining overlap.
Use suitably recent external holdouts when investigating generalization.

This first version freezes Maia, has no LoRA stage, no search, no automatic
training resume, no live platform integration, and no browser gameplay automation.
