# Chess persona modeling

Game preparation, frozen Maia adaptation, and move-conditioned timing

Source snapshot: [1bc429b5517280819910fb573b0593a770f0a904](https://github.com/aspetzle-dotcom/Maia-Chess/tree/1bc429b5517280819910fb573b0593a770f0a904)

Date: 21 October 2026

## 2. Pipeline and current status

Download player game archives and extract chronological pre-move examples.

Train separate player adapters on a frozen Maia-3 5M backbone.

Predict legal moves and move-conditioned time distributions.

The repository contains a runnable pipeline, not validated grandmaster personas.

Hikaru and Caruana profiles are experimental models inspired by recorded games. The architecture describes separate learned parameters for each player. There is no measured Elo or established style-fidelity result.

Sources:
- [README.md](https://github.com/aspetzle-dotcom/Maia-Chess/blob/1bc429b5517280819910fb573b0593a770f0a904/README.md)
- [docs/architecture.md](https://github.com/aspetzle-dotcom/Maia-Chess/blob/1bc429b5517280819910fb573b0593a770f0a904/docs/architecture.md)

## 3. Joint prediction task

Factor the joint distribution into move choice and time conditional on the move.

Condition both on board and historical context.

Train one independent checkpoint per player.

Retain a pre-move marginal timing prediction for honest evaluation.

```text
P(move,time | x,p) = P(move | x,p) P(time | move,x,p)
```

Knowing the actual move at evaluation gives a conditional timing score. Before a move occurs, timing must marginalize over possible moves. Those metrics answer different questions.

Sources:
- [docs/architecture.md](https://github.com/aspetzle-dotcom/Maia-Chess/blob/1bc429b5517280819910fb573b0593a770f0a904/docs/architecture.md)

## 4. Archive acquisition and caching

Fetch Chess.com monthly archives sequentially.

Cache completed months and refresh the current month.

Limit by date or number of months for an initial coverage audit.

Downloaded games and model checkpoints stay outside tracked source files.

The README recommends inspecting modest downloads before entire careers. Games without adequate clocks can still contribute move-policy examples. A clock-coverage audit is necessary before interpreting a profile as faster or slower.

Sources:
- [README.md](https://github.com/aspetzle-dotcom/Maia-Chess/blob/1bc429b5517280819910fb573b0593a770f0a904/README.md)

## 5. PGN preparation and exclusions

Extract eight-position board history and the last 16 plies.

Remove duplicates and audit coverage by player and time class.

Exclude unsupported variants, staged controls, correspondence games, and unknown dates.

Keep unavailable timing labels masked rather than inventing durations.

Inputs contain pre-move clocks and past move durations. Current-move labels, final game results, engine evaluations, and after-move clocks are excluded from model inputs.

Sources:
- [README.md](https://github.com/aspetzle-dotcom/Maia-Chess/blob/1bc429b5517280819910fb573b0593a770f0a904/README.md)
- [docs/architecture.md](https://github.com/aspetzle-dotcom/Maia-Chess/blob/1bc429b5517280819910fb573b0593a770f0a904/docs/architecture.md)

## 6. Chronological game-level splits

Split entire games chronologically into train, validation, and test.

Keep both sides of a game together, including games between target players.

Keep equal timestamps in the same split.

Filter to the target player before training or sampling.

Validation is uncapped. Optional training caps use seeded reservoir sampling over the file, while the default uses all training positions shuffled without replacement. Missing player coverage is an error.

Sources:
- [docs/architecture.md](https://github.com/aspetzle-dotcom/Maia-Chess/blob/1bc429b5517280819910fb573b0593a770f0a904/docs/architecture.md)

## 7. Clock-consumption labels

For a single-stage Fischer increment, infer elapsed clock consumption from adjacent same-side clocks.

The first move by each side has no timing label.

A missing clock breaks the next same-side subtraction.

Mask negative or over-budget durations.

```text
duration = clock_before + increment − clock_after
```

This measures server-recorded clock consumption, including platform effects. It cannot recover private thought time, network latency, or premove intent. A later clock must not be subtracted from a stale earlier clock.

Sources:
- [docs/architecture.md](https://github.com/aspetzle-dotcom/Maia-Chess/blob/1bc429b5517280819910fb573b0593a770f0a904/docs/architecture.md)

## 8. Relative-duration bins

Normalize duration by the clock available before the move.

Use 32 bins from near-instant play to the remaining budget.

The first interval is [0, 0.0001), followed by logarithmic intervals up to one.

Bin centers approximate means and medians in seconds.

```text
relative duration = duration / clock_before
```

The first bin uses an arithmetic midpoint and later bins use geometric centers for approximate means. This representation does not model within-bin timing or flagging behavior. Coarse bin boundaries can overcover an interval.

Sources:
- [docs/architecture.md](https://github.com/aspetzle-dotcom/Maia-Chess/blob/1bc429b5517280819910fb573b0593a770f0a904/docs/architecture.md)

## 9. Frozen Maia representations

Load the pinned Maia-3 5M implementation and compatible checkpoint.

Extract 64 square representations and legal candidate logits.

Keep backbone parameters frozen.

Use the upstream move vocabulary and tokenization, including promotions.

A temporary forward hook extracts transformer features. Weights load strictly with weights_only=True. Maia’s original ponder head is present but is not the persona timing distribution.

Sources:
- [docs/architecture.md](https://github.com/aspetzle-dotcom/Maia-Chess/blob/1bc429b5517280819910fb573b0593a770f0a904/docs/architecture.md)
- [README.md](https://github.com/aspetzle-dotcom/Maia-Chess/blob/1bc429b5517280819910fb573b0593a770f0a904/README.md)

## 10. History and context encoding

Encode source and destination squares, promotion, side, past duration, and missingness.

A 64-dimensional GRU summarizes the last 16 plies.

Fuse pooled board features with 15 numeric context features and a 32-dimensional player embedding.

Project the combined context to 128 dimensions.

Square embeddings use 16 dimensions each and promotion embeddings use eight. Black moves use python-chess square mirroring. Numeric context includes both clocks, time control, increment, and ratings.

Sources:
- [docs/architecture.md](https://github.com/aspetzle-dotcom/Maia-Chess/blob/1bc429b5517280819910fb573b0593a770f0a904/docs/architecture.md)

## 11. Player adapter and legal-move scorer

Use a residual adapter with dimensions 128, 32, and 128.

Combine adapted context with each legal move’s square and promotion features.

Predict a scalar correction to its base policy logit.

Initialize the policy correction to zero.

```text
P(move | x,p) = softmax(base logits + learned correction)
```

Each player has independent trainable layers and optimizer state. Starting from zero correction preserves the base policy initially. Legal candidate scoring avoids allocating probability to illegal actions.

Sources:
- [docs/architecture.md](https://github.com/aspetzle-dotcom/Maia-Chess/blob/1bc429b5517280819910fb573b0593a770f0a904/docs/architecture.md)

## 12. Move-conditioned timing head

Project each candidate representation through 128, 64, and 32 output dimensions.

Softmax produces a distribution over relative-duration bins.

Training conditions timing loss on the observed move when a valid label exists.

Policy learning can use positions without timing labels.

The conditional head permits different time distributions for different candidate moves. It is not the backbone’s ponder head and not a deterministic move-duration regressor.

Sources:
- [docs/architecture.md](https://github.com/aspetzle-dotcom/Maia-Chess/blob/1bc429b5517280819910fb573b0593a770f0a904/docs/architecture.md)

## 13. Training objective

Use legal-move cross-entropy and masked conditional timing cross-entropy.

Regularize the adapted policy toward the frozen base distribution.

Default weights are 0.5 for timing and 0.02 for KL.

AdamW uses learning rate 0.0003 and gradient clipping 1.0.

```text
L = CE_move + 0.5 CE_time + 0.02 KL(base ‖ adapted)
```

These are initial settings, not tuned scientific findings. CUDA uses mixed precision while CPU tests use float32. The default epoch count is three, while example workflows may explicitly request ten.

Sources:
- [docs/architecture.md](https://github.com/aspetzle-dotcom/Maia-Chess/blob/1bc429b5517280819910fb573b0593a770f0a904/docs/architecture.md)

## 14. Budgeted training and checkpoints

An explicit update budget overrides epoch count.

Validate at fixed successful-update intervals and at the final update.

best.pt uses combined move and marginal-timing validation score.

best-move.pt uses move NLL and last.pt preserves the completed budget.

The corrected example uses 3,130 updates, validation every 250, batch size 32, learning rate 0.0003, and seed 42. An explicit update budget is not shortened by early stopping. These jobs are fresh runs, not resumable training.

Sources:
- [README.md](https://github.com/aspetzle-dotcom/Maia-Chess/blob/1bc429b5517280819910fb573b0593a770f0a904/README.md)
- [docs/architecture.md](https://github.com/aspetzle-dotcom/Maia-Chess/blob/1bc429b5517280819910fb573b0593a770f0a904/docs/architecture.md)

## 15. Pre-move marginal timing

Weight every move-conditioned timing distribution by its predicted move probability.

Evaluate marginal timing NLL before the actual move is known.

Report conditional timing NLL separately.

Do not compare raw training objective directly with the different validation score.

```text
P(time | x,p) = Σ_move P(move | x,p) P(time | move,x,p)
```

The training timing CE uses the observed move, whereas the combined validation selection uses marginal timing. A fixed seeded training subset is evaluated with the same evaluation metrics to make training/validation comparisons more informative.

Sources:
- [docs/architecture.md](https://github.com/aspetzle-dotcom/Maia-Chess/blob/1bc429b5517280819910fb573b0593a770f0a904/docs/architecture.md)

## 16. Metrics and empirical baselines

Report move accuracy and NLL against the frozen backbone.

Report conditional and marginal timing NLL, approximate MAE, and interval coverage and width.

Group by time class, exact base+increment, and remaining-clock band.

Fit histogram and median-time baselines on training data only.

The nominal 80% interval expands to bin boundaries. Sparse control/clock groups fall back to the player baseline. Histograms use smoothing and median seconds are bounded by remaining clock. Save per-position game-identifiable validation records for audits.

Sources:
- [docs/architecture.md](https://github.com/aspetzle-dotcom/Maia-Chess/blob/1bc429b5517280819910fb573b0593a770f0a904/docs/architecture.md)

## 17. Execution and saved artifacts

download and prepare create audited processed splits.

train --player NAME produces the independent player checkpoint.

evaluate reads a checkpoint and writes test metrics.

Coverage, source hashes, selected offsets, batch-order hashes, and timing baselines support reproducibility.

Install -e ".[train,dev]" and requirements-maia.txt. A first run may fetch backbone weights from Hugging Face, while saved persona checkpoints embed them. The repository includes Colab notebooks and synthetic regression tests.

Sources:
- [README.md](https://github.com/aspetzle-dotcom/Maia-Chess/blob/1bc429b5517280819910fb573b0593a770f0a904/README.md)
- [docs/architecture.md](https://github.com/aspetzle-dotcom/Maia-Chess/blob/1bc429b5517280819910fb573b0593a770f0a904/docs/architecture.md)

## 18. Generalization and interpretation

Synthetic tests verify software behavior, not style fidelity or playing strength.

Different time controls and eras confound player-speed comparisons.

Maia pretraining may overlap older downstream evaluation games.

Reserve suitably recent external games for final confirmation.

The first version has no search, LoRA stage, automatic resume, or live platform integration. Previously inspected test results cannot become an untouched confirmation set. Caruana’s proposed slower profile remains unverified.

Sources:
- [docs/architecture.md](https://github.com/aspetzle-dotcom/Maia-Chess/blob/1bc429b5517280819910fb573b0593a770f0a904/docs/architecture.md)
- [README.md](https://github.com/aspetzle-dotcom/Maia-Chess/blob/1bc429b5517280819910fb573b0593a770f0a904/README.md)
