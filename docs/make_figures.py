"""
Regenerate the figure of `docs/zenodo-trajectory-repair.md`.

It compares the trajectories as published on Zenodo with the repaired ones, so it
has to run after `scripts/fix_zenodo_trajectories.py`, which leaves the originals
next to the repaired files under the suffix `.zenodo-orig.npy`.

    python docs/make_figures.py --datapath=data/
"""

import sys
from pathlib import Path

import fire
import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))

from fix_zenodo_trajectories import (  # noqa: E402
    BACKUP_SUFFIX,
    FACTOR,
    segment_length,
    steady_state_rate,
)

DATA = "training-trajectories/2D-delta-omega/"
NJUMPS = 48


def main(
    datapath: str = "data/",
    outdir: str = "docs/figures/",
    num_show: int = 1_000_000,
) -> None:
    """Draw the before/after comparison of the published 2D training set.

    Args:
        datapath (str): Root folder holding the downloaded Zenodo data.
        outdir (str): Folder the figure is written to.
        num_show (int): Number of trajectories to read. The full four million
            give the same picture, and cost four times the memory.
    """
    root = Path(datapath) / DATA
    taus_path = root / "taus-2D.npy"
    orig_path = taus_path.with_suffix(BACKUP_SUFFIX)
    for p in (taus_path, orig_path, root / "param_rand_list-2D.npy"):
        assert p.is_file(), (
            f"File not found {p}. The figure compares the published data with "
            f"the repaired one, so run scripts/fix_zenodo_trajectories.py first."
        )

    n = num_show
    params = np.asarray(
        np.load(root / "param_rand_list-2D.npy", mmap_mode="r")[:n], np.float64
    )
    before = np.asarray(np.load(orig_path, mmap_mode="r")[:n], np.float64)
    after = np.asarray(np.load(taus_path, mmap_mode="r")[:n], np.float64)

    delta, omega = params[:, 0], params[:, 1]
    rate = steady_state_rate(delta, omega, 1.0)
    tf = segment_length(delta, omega, NJUMPS, 1.0, FACTOR)

    fig, ax = plt.subplots(1, 2, figsize=(11.5, 4.2))

    # --- left: the delays of a TLS trajectory are i.i.d., so the mean delay
    # cannot depend on the jump index. The bug puts its corrupted entry at the
    # index where the continuation started, which piles up near the end.
    k = np.arange(NJUMPS)
    ax[0].plot(k, (before * rate[:, None]).mean(axis=0), "o-", ms=3, label="as published")
    ax[0].plot(k, (after * rate[:, None]).mean(axis=0), "s-", ms=3, label="repaired")
    ax[0].axhline(1.0, color="k", ls="--", lw=1, label=r"exact value $1$")
    ax[0].set(
        xlabel="jump index $k$",
        ylabel=r"$\langle \tau_k \rangle \, n_{\rm ss}$",
        title="Mean time delay per jump index\n"
        "(the delays are i.i.d., so this must be flat)",
    )
    ax[0].legend()

    # --- right: every jump of the first stretch happened before tf, so an
    # unaffected trajectory cannot have a total elapsed time beyond it.
    bins = np.linspace(0.2, 2.8, 200)
    ax[1].hist(before.sum(axis=1) / tf, bins=bins, alpha=0.55, label="as published")
    ax[1].hist(after.sum(axis=1) / tf, bins=bins, alpha=0.55, label="repaired")
    ax[1].axvline(1.0, color="k", ls="--", lw=1)
    ax[1].text(1.02, ax[1].get_ylim()[1] * 0.75, r"  one stretch, $t_f$", fontsize=9)
    ax[1].set(
        xlabel=r"total elapsed time $/\, t_f$",
        ylabel="trajectories",
        title="Total elapsed time of a 48-jump trajectory\n"
        r"(9.3% sit near $2\,t_f$ before the repair)",
    )
    ax[1].legend()

    fig.suptitle(
        "Zenodo 2D training trajectories: effect of the stitching bug "
        f"({n:,} of {len(np.load(taus_path, mmap_mode='r')):,} shown)",
        fontsize=11,
    )
    fig.tight_layout()

    out = Path(outdir)
    out.mkdir(parents=True, exist_ok=True)
    fig.savefig(out / "zenodo-trajectory-repair.png", dpi=140)
    plt.close(fig)
    print(f"wrote {(out / 'zenodo-trajectory-repair.png').resolve()}")
    return


if __name__ == "__main__":
    fire.Fire(main)
