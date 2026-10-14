# Architecture and data conventions

## Prediction task

For player p and pre-move context x, learn:

`P(move, duration | x, p) = P(move | x, p) P(duration | move, x, p)`.

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

Balanced sampling gives each player equal expected training weight. A global
chronological split keeps both sides of the same game together, including games
between the target players. Equal timestamps stay in the same split. Validation
selects the best checkpoint using a macro-average of player move NLL and weighted
marginal timing NLL. Test data is evaluated separately and never used by training.

Metrics include move accuracy and NLL for both the adapted policy and frozen
backbone. Timing metrics include conditional NLL (knowing the actual move),
pre-move marginal NLL, approximate mean absolute error in seconds and coverage of
a nominal 80% interval expanded to bin boundaries. The latter can overcover due
to coarse bins. Results are grouped by player and time class.

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
