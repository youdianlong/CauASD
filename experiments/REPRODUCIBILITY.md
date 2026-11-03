# Evaluation reproducibility

Run commands from the project root. Splits 1--3 use NTU-60 and splits 4--6 use NTU-120. Checkpoints must match the method and split; they are not included in the repository.

For each method and split, the checkpoint with the highest accuracy on the original, unwarped unseen-class test set during training is selected and kept fixed for Tables II--IV and S4. No checkpoint is reselected for an individual transformation type or factor. The `CHECKPOINT` argument below refers to this selected model.

## Protocols and commands

The following maps Tables II--IV and S4 to the released evaluation scripts and protocols used in this work.

| Paper table | Script in this directory | Selection | Output frames at sf=0.5 / 2.0 |
| --- | --- | --- | --- |
| II: uniform rescaling | [test_speed_warps.py](test_speed_warps.py) | `--warp linear` | 50 / 50 |
| III: random nonlinear warping | [test_speed_warps.py](test_speed_warps.py) | `--warp random` | 25 / 100 |
| IV: sinusoidal / quadratic warping | [test_speed_warps.py](test_speed_warps.py) | `--warp sinusoidal` / `quadratic` | 25 / 100 |
| S4: full-trajectory linear resampling, NTU-60 | [test_variable_length_speed.py](test_variable_length_speed.py) | no warp selector | 25 / 100 |

All inputs start from the preprocessed 50-frame test sequences. The uniform transform uses `tau[t] = clip(t / sf, 0, 49)`. Nonlinear modes instead use `T_out = max(2, round(50 * sf))` and sample the complete source interval [0, 49]. They are not fixed-length clipped transforms.

For nonlinear modes, let `u = j / (T_out - 1)` and `tau = 49 * q(u)`:

- Random: draw independent standard-normal values, set local increments to `exp(strength * noise)`, and normalize their cumulative sum to [0, 1].
- Sinusoidal: `q(u) = u + a * sin(2*pi*u) / (2*pi)`, where `a = clip(strength, 0, 0.8)`.
- Quadratic: choose `q(u) = u^2` or `1 - (1-u)^2` independently per sample with equal probability. The strength argument does not affect this mode.

Command template for one method and split; set the checkpoint to the trained model for that method and split:

```bash
SPLIT=1
METHOD=CauASD
CHECKPOINT=output/model/model.pt
WARP=linear

python experiments/test_speed_warps.py \
  --project-root . --split "$SPLIT" --checkpoint "$CHECKPOINT" \
  --warp "$WARP" --speed-factors 0.5,2.0 \
  --nonlinear-strength 0.5 --seed 2026 --batch-size 64 \
  --device cuda --gpu 0 \
  --output "analysis/reproduction/${METHOD}_split${SPLIT}_${WARP}.json"
```

Set `WARP` to `random`, `sinusoidal`, or `quadratic` for the corresponding mode. Evaluation uses seed 2026, batch size 64, and nonlinear strength 0.5 for random and sinusoidal warping; strength is not used by quadratic or uniform warping. Keep the seed, factor order, batch size, data order, and strength identical across methods: random draws depend on the factor index and batch starting index. This seed controls evaluation perturbations, not model training.

For S4, use NTU-60 splits 1--3 and the same method/split checkpoint:

```bash
python experiments/test_variable_length_speed.py \
  --project-root . --split "$SPLIT" --checkpoint "$CHECKPOINT" \
  --speed-factors 0.5,0.75,1.0,1.25,1.5,1.75,2.0 \
  --batch-size 64 --device cuda --gpu 0 \
  --output "analysis/reproduction/${METHOD}_split${SPLIT}_variable_length.json"
```

S4 uses `tau[j] = j * 49 / (T_out - 1)`; output lengths are 25, 38, 50, 62, 75, 88, and 100. Its temporal transform is deterministic and has no seed argument. This does not specify the training seed of the checkpoint. The command above does not enable prototype-guided DA.

Fixed-length and variable-length evaluations differ jointly in output length, sampling, and boundary handling. Their absolute accuracies do not isolate the effect of endpoint repetition.

## Class-level statistics (Table S1)

The corrected [class-level statistics](results/natural_speed/natural_speed_by_class_all_splits.csv) contain 45 class records across splits 1--6, using `valid_body_transition_v2`. They report class-specific q30/q70 values, observed rank cutoffs, stratum sample counts, and score summaries. Table S1 uses unweighted class means for threshold and score summaries, and sums the sample counts.

## Accuracy aggregation and Table V

Tables II--IV and S4 use **micro accuracy**: total correct predictions divided by total test samples within a split. They do not average batch accuracies. Dataset-level results use an equal-weight average of the three split accuracies.

[Table V's evaluator](natural_speed/run_natural_speed_gap_stgcn.py) uses original, unwarped inputs. Within each class, stable ascending score ranking assigns the lowest and highest `floor(0.30 * n_class)` samples to slower/faster strata; all remaining samples form the middle stratum. Invalid normalized scores are excluded before ranking.

For each stratum: compute accuracy within each unseen class, average equally over represented classes, then average equally over the three splits. No batch-level accuracy averaging is used. For each split, `Gap = abs(Accuracy_slower - Accuracy_faster)`; average these gaps across splits using unrounded values. This need not equal the absolute difference between the two averaged accuracies.

The corrected `valid_body_transition_v2` implementation masks valid bodies and adjacent-frame transitions separately. Changed scores, exclusions, and stratum membership can change Table V. Macro stratum accuracies in V are not intended to reconstruct S4's micro accuracies.

The following gaps are transcribed from the corrected six-split run outputs supplied for this revision. Units are percentage points; PGFA-aug denotes PGFA with temporal augmentation (PGFA-double-dagger in the paper).

| Dataset | Split | PGFA | PGFA-aug | CauASD |
| --- | --- | ---: | ---: | ---: |
| NTU-60 | 1 | 10.97 | 14.88 | 7.31 |
| NTU-60 | 2 | 1.20 | 1.00 | 2.20 |
| NTU-60 | 3 | 15.05 | 12.83 | 16.30 |
| NTU-60 | Mean | 9.07 | 9.57 | 8.60 |
| NTU-120 | 4 | 12.93 | 5.57 | 16.26 |
| NTU-120 | 5 | 11.97 | 10.54 | 8.83 |
| NTU-120 | 6 | 9.54 | 9.17 | 6.29 |
| NTU-120 | Mean | 11.48 | 8.43 | 10.46 |

Table V runs save CSV metrics, JSON summaries, and NPZ arrays containing labels, original sample indices, scores, and aligned predictions for all three methods. Use the NumPy-only [paired postprocessor](natural_speed/summarize_paired_uncertainty.py) without training or model inference:

```bash
python experiments/natural_speed/summarize_paired_uncertainty.py \
  --predictions-dir analysis/natural_speed_v2 \
  --data-root data/zeroshot \
  --output-dir analysis/reviewer5_paired
```

This requires six corrected NPZ/JSON pairs and each split's `unseen_sample_names.npy` and `unseen_label.npy`. Saved metrics and sample alignment are checked before bootstrapping.

### Paired uncertainty results

Results below are transcribed from the completed server run. Differences are CauASD minus baseline, in percentage points; brackets contain percentile 95% confidence intervals.

| Dataset | Baseline | Gap difference [95% CI] | Faster accuracy difference [95% CI] |
| --- | --- | ---: | ---: |
| NTU-60 | PGFA | -0.47 [-2.91, 2.22] | 3.27 [-0.33, 4.92] |
| NTU-60 | PGFA-aug | -0.96 [-3.82, 2.16] | 4.49 [1.78, 6.66] |
| NTU-120 | PGFA | -1.02 [-3.21, 2.25] | 1.67 [-1.24, 3.57] |
| NTU-120 | PGFA-aug | 2.04 [-0.62, 5.54] | 2.75 [0.77, 4.51] |

We use 2,000 paired subject-cluster bootstrap draws (seed 2026), with 20 subjects on NTU-60 and 53 on NTU-120. Subject multiplicities are shared across methods and across the three splits within each dataset; strata remain fixed. No draws were rejected for empty class-strata in this run.

Negative Gap differences and positive Faster-accuracy differences favor CauASD. Faster accuracy relative to PGFA-aug has intervals above zero on both datasets; all Gap-difference intervals include zero. Intervals are unadjusted for multiple comparisons and conditional on fixed checkpoints, splits, exclusions, and strata; these confidence intervals quantify test-subject sampling uncertainty for the fixed checkpoints, class splits, and speed strata.
