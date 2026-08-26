"""
Build the embedding-comparison report for the 2D NPE posterior from the four
runs of `scripts/npe_2d_sbi.py` (--embedding=none/deepset/hist/cnn) on a
95%/5% train/held-out split of the full Zenodo 2D dataset.

Reads each run's `summary.json` and figures from
`--run_root` (default: the shared location used by the plan that produced
them, `data/models/npe-sbi-2D/embedding-comparison-95-5/` in the main
checkout) and writes:

  - the comparison figures, copied and renamed under `--figures_dir`
    (default `docs/figures/`)
  - the report itself, `--out` (default `docs/npe-2d-embedding-comparison.md`)

Usage
-----
    python docs/make_npe_embedding_report.py
"""

import json
import shutil
from pathlib import Path
from typing import Any, Dict

import fire

EMBEDDINGS = ["none", "deepset", "hist", "cnn"]
PARAM_LABELS = ["Δ", "Ω"]

EMBEDDING_LABEL = {
    "none": "No embedding (raw trajectory)",
    "deepset": "DeepSets (learned permutation-invariant summary)",
    "hist": "Hist-Dense (fixed histogram bins, paper architecture)",
    "cnn": "CNN (1D convolutions, not permutation invariant — control)",
}

DEFAULT_RUN_ROOT = (
    "/Users/enrico.rinaldi/Projects/ParamEst-NN/data/models/npe-sbi-2D/"
    "embedding-comparison-95-5"
)


def _load_summaries(run_root: Path) -> Dict[str, Dict[str, Any]]:
    summaries = {}
    for embedding in EMBEDDINGS:
        path = run_root / embedding / "summary.json"
        assert path.is_file(), f"Missing {path} -- run scripts/npe_2d_sbi.py first"
        summaries[embedding] = json.loads(path.read_text())
    return summaries


def _copy_figures(run_root: Path, figures_dir: Path) -> None:
    figures_dir.mkdir(parents=True, exist_ok=True)
    shutil.copy(
        run_root / "none" / "training_data.png",
        figures_dir / "npe-2d-training-data.png",
    )
    figure_names = {
        "posterior_observation.png": "posterior-observation",
        "sbc_rank_histogram.png": "sbc-rank-histogram",
        "expected_coverage.png": "expected-coverage",
        "tarp.png": "tarp",
    }
    for embedding in EMBEDDINGS:
        for src_name, dst_stem in figure_names.items():
            src = run_root / embedding / src_name
            assert src.is_file(), f"Missing {src}"
            shutil.copy(src, figures_dir / f"npe-2d-{embedding}-{dst_stem}.png")


def _config_table(summaries: Dict[str, Dict[str, Any]]) -> str:
    header = (
        "| embedding | flow | hidden features | transforms | "
        "embedding-specific | z-score(x) |\n"
        "| --- | --- | --- | --- | --- | --- |\n"
    )
    rows = []
    for embedding in EMBEDDINGS:
        cfg = summaries[embedding]["config"]
        if embedding == "hist":
            extra = f"nbins={cfg['hist_nbins']}, taumax={cfg['hist_taumax']}"
        elif embedding == "cnn":
            extra = f"kernel_size={cfg['cnn_kernel_size']}, output_dim={cfg['embedding_output_dim']}"
        elif embedding == "deepset":
            extra = f"output_dim={cfg['embedding_output_dim']}"
        else:
            extra = "--"
        rows.append(
            f"| `{embedding}` | {cfg['model']} | {cfg['hidden_features']} | "
            f"{cfg['num_transforms']} | {extra} | {cfg['z_score_x']} |"
        )
    return header + "\n".join(rows)


def _training_table(summaries: Dict[str, Dict[str, Any]]) -> str:
    header = (
        "| embedding | epochs trained | best validation loss (lower is better) |\n"
        "| --- | --- | --- |\n"
    )
    rows = [
        f"| `{e}` | {summaries[e]['training']['epochs']} | "
        f"{summaries[e]['training']['best_validation_loss']:.4f} |"
        for e in EMBEDDINGS
    ]
    return header + "\n".join(rows)


def _convergence_note(summaries: Dict[str, Dict[str, Any]]) -> str:
    capped = [
        e for e in EMBEDDINGS
        if summaries[e]["training"]["epochs"] >= summaries[e]["config"]["max_num_epochs"]
    ]
    if not capped:
        return ""
    which = (
        "All four" if len(capped) == len(EMBEDDINGS)
        else f"{len(capped)} of {len(EMBEDDINGS)} (" + ", ".join(f"`{e}`" for e in capped) + ")"
    )
    return (
        f"{which} runs hit the `max_num_epochs` cap rather than stopping "
        "early on a validation-loss plateau (`scripts/npe_2d_sbi.py` prints "
        '"Maximum number of epochs reached, but network has not yet fully '
        'converged" for each). The validation-loss ranking above reflects '
        "where training happened to be when the shared epoch budget ran "
        "out, not a converged comparison -- treat \"best validation loss\" "
        "as directional, not final."
    )


def _observation_table(summaries: Dict[str, Dict[str, Any]]) -> str:
    true_theta = summaries["none"]["observation"]["true"]
    header = (
        f"True (Δ, Ω) = ({true_theta[0]:.4f}, {true_theta[1]:.4f})\n\n"
        "| embedding | posterior mean Δ | posterior std Δ | "
        "posterior mean Ω | posterior std Ω |\n"
        "| --- | --- | --- | --- | --- |\n"
    )
    rows = []
    for e in EMBEDDINGS:
        obs = summaries[e]["observation"]
        rows.append(
            f"| `{e}` | {obs['posterior_mean'][0]:.4f} | {obs['posterior_std'][0]:.4f} | "
            f"{obs['posterior_mean'][1]:.4f} | {obs['posterior_std'][1]:.4f} |"
        )
    return header + "\n".join(rows)


def _calibration_table(summaries: Dict[str, Dict[str, Any]]) -> str:
    header = (
        "| embedding | SBC KS p(Δ) | SBC KS p(Ω) | C2ST rank(Δ) | C2ST rank(Ω) | "
        "TARP ATC | TARP KS p-value |\n"
        "| --- | --- | --- | --- | --- | --- | --- |\n"
    )
    rows = []
    for e in EMBEDDINGS:
        diag = summaries[e]["diagnostics"]
        sbc = diag["sbc"]
        rows.append(
            f"| `{e}` | {sbc['ks_pvals'][0]:.4f} | {sbc['ks_pvals'][1]:.4f} | "
            f"{sbc['c2st_ranks'][0]:.4f} | {sbc['c2st_ranks'][1]:.4f} | "
            f"{diag['tarp']['atc']:.4f} | {diag['tarp']['ks_pval']:.4f} |"
        )
    return header + "\n".join(rows)


def _calibration_notes(summaries: Dict[str, Dict[str, Any]]) -> str:
    """A data-driven paragraph on how to read the marginal-vs-joint checks
    together, instead of leaving the raw table to speak for itself. Every
    number here is computed from `summaries`, not asserted -- if a future
    rerun's numbers differ, this paragraph changes with them.
    """
    flagged = []
    passed = []
    for e in EMBEDDINGS:
        sbc = summaries[e]["diagnostics"]["sbc"]
        bad = [PARAM_LABELS[i] for i, p in enumerate(sbc["ks_pvals"]) if p < 0.05]
        if bad:
            flagged.append(f"`{e}` ({', '.join(bad)})")
        else:
            passed.append(f"`{e}`")
    if not flagged:
        return (
            "Every embedding passes the SBC rank-uniformity check (p >= 0.05) "
            "on both parameters, consistent with TARP's clean joint result "
            "above."
        )
    tarp_ks = [summaries[e]["diagnostics"]["tarp"]["ks_pval"] for e in EMBEDDINGS]
    tarp_atcs = [abs(summaries[e]["diagnostics"]["tarp"]["atc"]) for e in EMBEDDINGS]
    worst = min(
        (
            (e, i, p)
            for e in EMBEDDINGS
            for i, p in enumerate(summaries[e]["diagnostics"]["sbc"]["ks_pvals"])
        ),
        key=lambda t: t[2],
    )
    if len(flagged) == len(EMBEDDINGS):
        lead = (
            "Every run rejects SBC rank-uniformity (p < 0.05) on at least "
            f"one marginal: {', '.join(flagged)}."
        )
    else:
        lead = (
            f"{len(flagged)} of {len(EMBEDDINGS)} runs reject SBC "
            f"rank-uniformity (p < 0.05) on at least one marginal: "
            f"{', '.join(flagged)}; {', '.join(passed)} pass"
            f"{'es' if len(passed) == 1 else ''} both parameters."
        )
    if all(k >= 0.05 for k in tarp_ks):
        tarp_sentence = (
            "TARP -- the joint check that `scripts/npe_2d_sbi.py`'s own "
            "docstring calls \"necessary and sufficient\", unlike the "
            f"marginal SBC test -- stays clean for all {len(EMBEDDINGS)} "
            f"runs (ATC between {min(tarp_atcs):.4f} and "
            f"{max(tarp_atcs):.4f}, KS p-value at or above 0.05 in every "
            "case), so none of these marginal rejections corresponds to a "
            "joint-posterior calibration failure severe enough for TARP to "
            "catch."
        )
    else:
        failing = [e for e, k in zip(EMBEDDINGS, tarp_ks) if k < 0.05]
        tarp_sentence = (
            "TARP -- the joint, \"necessary and sufficient\" check per "
            "`scripts/npe_2d_sbi.py`'s own docstring -- also rejects for "
            f"{', '.join(f'`{e}`' for e in failing)}, so at least one of "
            "these marginal SBC rejections corresponds to a genuine "
            "joint-posterior calibration problem, not just marginal noise."
        )
    return (
        f"{lead} {tarp_sentence} The sharpest single marginal rejection is "
        f"`{worst[0]}` on {PARAM_LABELS[worst[1]]} (p={worst[2]:.4f})."
    )


def _observation_bias_note(summaries: Dict[str, Dict[str, Any]]) -> str:
    """Reports how far each embedding's posterior mean at the single held-out
    example sits from the truth, in posterior standard deviations rather than
    raw units (a small absolute offset can be many sigma in a tight posterior,
    and a large one unremarkable in a wide one) -- and says plainly when that
    offset is ordinary single-draw scatter (|z| < 2) rather than a real
    discrepancy, since this is one example, not the calibration check.
    """
    true_theta = summaries["none"]["observation"]["true"]
    notes = []
    for i, label in enumerate(PARAM_LABELS):
        zs = [
            (summaries[e]["observation"]["posterior_mean"][i] - true_theta[i])
            / summaries[e]["observation"]["posterior_std"][i]
            for e in EMBEDDINGS
        ]
        same_direction = all(z > 0 for z in zs) or all(z < 0 for z in zs)
        if not same_direction:
            continue
        direction = "above" if zs[0] > 0 else "below"
        severity = (
            "notable" if all(abs(z) > 2.0 for z in zs)
            else "ordinary single-draw scatter, not a systematic issue"
        )
        notes.append(
            f"every embedding's posterior mean for {label} sits "
            f"{min(abs(z) for z in zs):.1f}-{max(abs(z) for z in zs):.1f} "
            f"posterior standard deviations {direction} the true value "
            f"({true_theta[i]:.4f}) at this one held-out observation -- "
            f"{severity}"
        )
    if not notes:
        return ""
    num_diagnostic = summaries["none"]["config"]["num_diagnostic"]
    return (
        "At this single held-out example, " + "; and ".join(notes) + ". "
        f"The calibration checks above run on {num_diagnostic:,} *different* "
        "held-out pairs and are what actually validates each posterior -- "
        "this one-example note is provided for context, not as a "
        "calibration result."
    )


def _best_by_tarp_atc(summaries: Dict[str, Dict[str, Any]]) -> str:
    return min(EMBEDDINGS, key=lambda e: abs(summaries[e]["diagnostics"]["tarp"]["atc"]))


def _best_by_validation_loss(summaries: Dict[str, Dict[str, Any]]) -> str:
    return min(EMBEDDINGS, key=lambda e: summaries[e]["training"]["best_validation_loss"])


def _zscore_footnote(summaries: Dict[str, Dict[str, Any]]) -> str:
    overridden = [
        e for e in EMBEDDINGS
        if summaries[e]["config"]["z_score_x"] != summaries[EMBEDDINGS[0]]["config"]["z_score_x"]
    ]
    if not overridden:
        return ""
    return (
        f"`{'`, `'.join(overridden)}` shows a different `z_score_x` than the "
        "others because its embedding needs the raw, physical-unit time "
        'delays -- `scripts/npe_2d_sbi.py` forces `z_score_x="none"` for '
        "it automatically (its histogram bin edges are fixed in physical "
        "units and would be broken by standardising x first). Every other "
        "hyperparameter, including the flow itself, is identical across "
        "all four runs."
    )


def main(
    run_root: str = DEFAULT_RUN_ROOT,
    figures_dir: str = "docs/figures",
    out: str = "docs/npe-2d-embedding-comparison.md",
) -> None:
    """Render the embedding-comparison report.

    Args:
        run_root (str): Directory holding the four `{none,deepset,hist,cnn}/`
            run outputs produced by `scripts/npe_2d_sbi.py`.
        figures_dir (str): Where the renamed comparison figures are written.
        out (str): Path of the Markdown report to write.
    """
    run_root_path = Path(run_root)
    summaries = _load_summaries(run_root_path)
    _copy_figures(run_root_path, Path(figures_dir))

    num_train = summaries["none"]["config"]["num_train"]
    num_diagnostic = summaries["none"]["config"]["num_diagnostic"]
    total = round(num_train / 0.95)  # num_train is 95% of the total by construction
    best_loss = _best_by_validation_loss(summaries)
    best_tarp = _best_by_tarp_atc(summaries)

    report = f"""# Comparing trajectory embeddings for the 2D NPE posterior

Four Neural Posterior Estimation runs of
[`scripts/npe_2d_sbi.py`](../scripts/npe_2d_sbi.py) on the same 95%/5%
train/held-out split of the full 2D `(Δ, Ω)` Zenodo dataset
({total:,} prior-draw/trajectory pairs total, {num_train:,} used for
training and the remaining {total - num_train:,} never seen during
training), differing only in `--embedding`. Every other hyperparameter
(flow family, hidden width, number of transforms, batch size, learning rate,
early-stopping patience, random seed) is identical across the four runs, so
any difference in the tables below is attributable to the embedding choice
alone.

| embedding | what it is |
| --- | --- |
| `none` | {EMBEDDING_LABEL['none']} |
| `deepset` | {EMBEDDING_LABEL['deepset']} |
| `hist` | {EMBEDDING_LABEL['hist']} |
| `cnn` | {EMBEDDING_LABEL['cnn']} |

![Training data overview](figures/npe-2d-training-data.png)

## Run configuration

{_config_table(summaries)}

{_zscore_footnote(summaries)}

## Training outcome

{_training_table(summaries)}

Best validation loss (negative log-likelihood on the held-out validation
split used internally by `NPE.train`) is lowest for `{best_loss}`.

{_convergence_note(summaries)}

## Posterior at a held-out example observation

{num_diagnostic:,} pairs from the held-out 5% were used for the calibration
checks below; one further pair from that same held-out pool, never used in
training or in the calibration checks, is the example observation here.

{_observation_table(summaries)}

{_observation_bias_note(summaries)}

## Calibration: SBC, expected coverage and TARP

Computed on {num_diagnostic:,} held-out (prior draw, prior predictive) pairs
per embedding, following `run_diagnostics` in `scripts/npe_2d_sbi.py`. A
well-calibrated posterior has SBC/TARP KS p-values that are not small
(the flat-rank null is not rejected), C2ST ranks near 0.5 (chance level), and
a TARP ATC (area between the empirical coverage curve and the diagonal)
close to 0.

{_calibration_table(summaries)}

{_calibration_notes(summaries)} By TARP ATC, `{best_tarp}` is the
best-calibrated of the four.

## Per-embedding figures

"""
    for e in EMBEDDINGS:
        report += f"""### `{e}` — {EMBEDDING_LABEL[e]}

![{e} posterior at the held-out observation](figures/npe-2d-{e}-posterior-observation.png)

![{e} SBC rank histogram](figures/npe-2d-{e}-sbc-rank-histogram.png)

![{e} expected coverage](figures/npe-2d-{e}-expected-coverage.png)

![{e} TARP](figures/npe-2d-{e}-tarp.png)

"""

    report += f"""## Reproducing

```bash
python docs/make_npe_embedding_report.py
```

regenerates this file and the figures above from the four runs'
`summary.json` files under `{run_root}`, once they exist (see
`docs/superpowers/plans/2026-08-25-npe-2d-embedding-comparison.md` for the
exact commands that produce them).
"""

    out_path = Path(out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(report)
    print(f"wrote {out_path.resolve()}")


if __name__ == "__main__":
    fire.Fire(main)
