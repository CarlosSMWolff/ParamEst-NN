# Comparing trajectory embeddings for the 2D NPE posterior

Four Neural Posterior Estimation runs of
[`scripts/npe_2d_sbi.py`](../scripts/npe_2d_sbi.py) on the same 95%/5%
train/held-out split of the full 2D `(Δ, Ω)` Zenodo dataset
(4,000,000 prior-draw/trajectory pairs total, 3,800,000 used for
training and the remaining 200,000 never seen during
training), differing only in `--embedding`. Every other hyperparameter
(flow family, hidden width, number of transforms, batch size, learning rate,
early-stopping patience, random seed) is identical across the four runs, so
any difference in the tables below is attributable to the embedding choice
alone.

| embedding | what it is |
| --- | --- |
| `none` | No embedding (raw trajectory) |
| `deepset` | DeepSets (learned permutation-invariant summary) |
| `hist` | Hist-Dense (fixed histogram bins, paper architecture) |
| `cnn` | CNN (1D convolutions, not permutation invariant — control) |

![Training data overview](figures/npe-2d-training-data.png)

## Run configuration

| embedding | flow | hidden features | transforms | embedding-specific | z-score(x) |
| --- | --- | --- | --- | --- | --- |
| `none` | zuko_nsf | 128 | 3 | -- | structured |
| `deepset` | zuko_nsf | 128 | 3 | output_dim=16 | structured |
| `hist` | zuko_nsf | 128 | 3 | nbins=700, taumax=100.0 | none |
| `cnn` | zuko_nsf | 128 | 3 | kernel_size=5, output_dim=16 | structured |

## Training outcome

| embedding | epochs trained | best validation loss (lower is better) |
| --- | --- | --- |
| `none` | 51 | 0.7482 |
| `deepset` | 51 | -0.2452 |
| `hist` | 51 | -0.7101 |
| `cnn` | 51 | 0.4564 |

Best validation loss (negative log-likelihood on the held-out validation
split used internally by `NPE.train`) is lowest for `hist`.

## Posterior at a held-out example observation

1,000 pairs from the held-out 5% were used for the calibration
checks below; one further pair from that same held-out pool, never used in
training or in the calibration checks, is the example observation here.

True (Δ, Ω) = (0.2047, 3.9033)

| embedding | posterior mean Δ | posterior std Δ | posterior mean Ω | posterior std Ω |
| --- | --- | --- | --- | --- |
| `none` | 0.9771 | 0.6738 | 3.5690 | 0.1957 |
| `deepset` | 1.0525 | 0.6810 | 3.8834 | 0.0843 |
| `hist` | 1.2101 | 0.7472 | 3.8638 | 0.0878 |
| `cnn` | 1.0580 | 0.7010 | 3.8640 | 0.1710 |

Every embedding's posterior mean for Δ overshoots the true value (0.2047) by at least 0.3 at this particular held-out observation, regardless of the embedding used. Since the calibration checks above run on 1,000 *different* held-out pairs, this shared bias looks like a property of this one example rather than a shortcoming of any particular embedding.

## Calibration: SBC, expected coverage and TARP

Computed on 1,000 held-out (prior draw, prior predictive) pairs
per embedding, following `run_diagnostics` in `scripts/npe_2d_sbi.py`. A
well-calibrated posterior has SBC/TARP KS p-values that are not small
(the flat-rank null is not rejected), C2ST ranks near 0.5 (chance level), and
a TARP ATC (area between the empirical coverage curve and the diagonal)
close to 0.

| embedding | SBC KS p(Δ) | SBC KS p(Ω) | C2ST rank(Δ) | C2ST rank(Ω) | TARP ATC | TARP KS p-value |
| --- | --- | --- | --- | --- | --- | --- |
| `none` | 0.5003 | 0.0106 | 0.5750 | 0.5745 | 0.0017 | 1.0000 |
| `deepset` | 0.8963 | 0.3627 | 0.5650 | 0.5960 | 0.0013 | 1.0000 |
| `hist` | 0.0404 | 0.0571 | 0.5575 | 0.5580 | 0.0045 | 1.0000 |
| `cnn` | 0.0036 | 0.4517 | 0.5805 | 0.5615 | 0.0030 | 1.0000 |

3 of 4 runs reject SBC rank-uniformity (p < 0.05) on at least one marginal: `none` (Ω), `hist` (Δ), `cnn` (Δ); `deepset` passes both parameters. TARP -- the joint check that `scripts/npe_2d_sbi.py`'s own docstring calls "necessary and sufficient", unlike the marginal SBC test -- stays clean for all four runs (ATC between 0.0013 and 0.0045, KS p-value near 1.0 in every case), so none of these marginal rejections corresponds to a joint-posterior calibration failure severe enough for TARP to catch. The sharpest single rejection is `cnn` on Δ (p=0.0036). By TARP ATC, `deepset` is the
best-calibrated of the four.

## Per-embedding figures

### `none` — No embedding (raw trajectory)

![none posterior at the held-out observation](figures/npe-2d-none-posterior-observation.png)
![none SBC rank histogram](figures/npe-2d-none-sbc-rank-histogram.png)
![none expected coverage](figures/npe-2d-none-expected-coverage.png)
![none TARP](figures/npe-2d-none-tarp.png)

### `deepset` — DeepSets (learned permutation-invariant summary)

![deepset posterior at the held-out observation](figures/npe-2d-deepset-posterior-observation.png)
![deepset SBC rank histogram](figures/npe-2d-deepset-sbc-rank-histogram.png)
![deepset expected coverage](figures/npe-2d-deepset-expected-coverage.png)
![deepset TARP](figures/npe-2d-deepset-tarp.png)

### `hist` — Hist-Dense (fixed histogram bins, paper architecture)

![hist posterior at the held-out observation](figures/npe-2d-hist-posterior-observation.png)
![hist SBC rank histogram](figures/npe-2d-hist-sbc-rank-histogram.png)
![hist expected coverage](figures/npe-2d-hist-expected-coverage.png)
![hist TARP](figures/npe-2d-hist-tarp.png)

### `cnn` — CNN (1D convolutions, not permutation invariant — control)

![cnn posterior at the held-out observation](figures/npe-2d-cnn-posterior-observation.png)
![cnn SBC rank histogram](figures/npe-2d-cnn-sbc-rank-histogram.png)
![cnn expected coverage](figures/npe-2d-cnn-expected-coverage.png)
![cnn TARP](figures/npe-2d-cnn-tarp.png)

## Reproducing

```bash
python docs/make_npe_embedding_report.py
```

regenerates this file and the figures above from the four runs'
`summary.json` files under `/Users/enrico.rinaldi/Projects/ParamEst-NN/data/models/npe-sbi-2D/embedding-comparison-95-5`, once they exist (see
`docs/superpowers/plans/2026-08-25-npe-2d-embedding-comparison.md` for the
exact commands that produce them).
