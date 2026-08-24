"""
Neural Posterior Estimation (NPE) on photon-counting trajectories for the 2D
parameter space (Delta, Omega), written directly against the `sbi` package.

This script is the command line version of the notebook `notebooks/4-NPE.ipynb`.
Both used to be written against `lampe`, whose development has stopped in favour
of `sbi`; the mapping between the two libraries is:

    lampe                                  ->  sbi
    ------------------------------------------------------------------------
    lampe.data.H5Dataset                   ->  NPE.append_simulations(theta, x)
    lampe.inference.NPE (zuko flow)        ->  sbi.inference.NPE + posterior_nn
    lampe.inference.NPELoss + GDStep       ->  NPE.train(...)
    manual preprocess/postprocess of theta ->  z_score_theta / prior transform
    estimator.flow(x).sample(...)          ->  posterior.sample((n,), x=x_o)
    lampe.plots.corner / mark_point        ->  sbi.analysis.pairplot(points=...)
    lampe.diagnostics.expected_coverage_mc ->  sbi.diagnostics.run_sbc + run_tarp
    custom DeepSet embedding               ->  PermutationInvariantEmbedding

The training pairs are the ones produced by `1-Trajectories_generation.ipynb` and
distributed on Zenodo: `param_rand_list-2D.npy` holds parameters drawn from a
uniform prior, `taus-2D.npy` holds one simulated trajectory of `njumps` time
delays for each parameter pair. Because those pairs are exactly
(prior draw, prior predictive) samples, a held-out slice of them can be reused
directly for SBC, expected coverage and TARP without running the simulator again.

Usage
-----
    python scripts/npe_2d_sbi.py --help
    python scripts/npe_2d_sbi.py --num_train=512000 --max_num_epochs=50
    python scripts/npe_2d_sbi.py --embedding=deepset --model=zuko_nsf \
        --hidden_features=64 --max_num_epochs=15

Outputs (figures, the trained posterior and a JSON summary) are written to
`--outdir`, which defaults to `data/models/npe-sbi-2D/`.
"""

import json
import pickle
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import matplotlib

matplotlib.use("Agg")  # the script is meant to run headless, e.g. on a cluster

import fire  # noqa: E402
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import torch  # noqa: E402
from sbi.analysis import pairplot, plot_tarp, sbc_rank_plot  # noqa: E402
from sbi.diagnostics import check_sbc, check_tarp, run_sbc, run_tarp  # noqa: E402
from sbi.inference import NPE  # noqa: E402
from sbi.neural_nets import posterior_nn  # noqa: E402
from sbi.neural_nets.embedding_nets import (  # noqa: E402
    FCEmbedding,
    PermutationInvariantEmbedding,
)
from sbi.utils import BoxUniform  # noqa: E402

# Parameter names and plotting labels, in the order used in the data files
PARAMETERS = ["delta", "omega"]
LABELS = [r"$\Delta$", r"$\Omega$"]

# Physical parameter ranges used to generate the training data (the prior).
# The detuning delta lives in [0, 3] and the drive frequency omega in [0.25, 5].
DELTA_MIN, DELTA_MAX = 0.0, 3.0
OMEGA_MIN, OMEGA_MAX = 0.25, 5.0

# Relative paths of the training data inside `datapath`
PARAMS_FILE = "2D-delta-omega/param_rand_list-2D.npy"
TAUS_FILE = "2D-delta-omega/taus-2D.npy"

# Density estimators that support mapping a bounded theta to an unconstrained
# space, which is the `sbi` counterpart of the manual [-1, 1] rescaling of theta
# done in the notebook, and which also removes NPE leakage outside the prior box.
UNCONSTRAINED_OK = ("mdn", "zuko_")


#############
#  GET DATA #
#############
def get_training_pairs(datapath: str) -> Tuple[np.ndarray, np.ndarray]:
    """Load the (parameters, trajectories) pairs used to train the NPE.

    Args:
        datapath (str): The root folder holding the downloaded Zenodo data.

    Returns:
        tuple(np.ndarray, np.ndarray): A tuple of 2 arrays (params, taus);
        the parameters first, with shape (num_pairs, 2), and the trajectories
        second, with shape (num_pairs, njumps).
    """
    fp = Path(datapath) / PARAMS_FILE
    ft = Path(datapath) / TAUS_FILE
    assert fp.is_file(), f"File not found {fp}"
    assert ft.is_file(), f"File not found {ft}"
    params = np.load(fp)
    taus = np.load(ft)
    assert len(params) == len(taus), "params and taus must have the same length"
    return params, taus


def plot_training_data(params: np.ndarray, taus: np.ndarray, outdir: Path) -> None:
    """Reproduce the exploratory plots of the notebook: a scatter plot of the
    sampled parameters, a few trajectories, and the histogram of a single one.

    Args:
        params (np.ndarray): The array of parameter pairs.
        taus (np.ndarray): The array of trajectories.
        outdir (Path): Where the figures are written.
    """
    fig, axes = plt.subplots(1, 3, figsize=(13.5, 4.0))

    axes[0].scatter(params[:1000, 0], params[:1000, 1], s=4)
    axes[0].set(xlabel=LABELS[0], ylabel=LABELS[1], title="Params scatter plot")

    for element in taus[:3]:
        axes[1].plot(element)
    axes[1].set(xlabel="Jump index", ylabel=r"$\tau$", title="3 trajectories")

    axes[2].hist(taus[0], bins=5)
    axes[2].set(xlabel=r"$\tau$", ylabel="Frequency", title="Single trajectory")

    fig.tight_layout()
    fig.savefig(outdir / "training_data.png", dpi=150)
    plt.close(fig)


############################
#  PRIOR AND NEURAL NETWORK #
############################
def build_prior(device: str = "cpu") -> BoxUniform:
    """Build the uniform prior over (delta, omega) used to generate the data.

    Args:
        device (str): The torch device the prior lives on.

    Returns:
        BoxUniform: The 2D uniform prior.
    """
    return BoxUniform(
        low=torch.tensor([DELTA_MIN, OMEGA_MIN]),
        high=torch.tensor([DELTA_MAX, OMEGA_MAX]),
        device=device,
    )


def build_embedding_net(
    embedding: str, embedding_output_dim: int, hidden_features: int
) -> Optional[torch.nn.Module]:
    """Build the embedding network that summarises a trajectory before it is fed
    to the flow.

    A trajectory is a set of time delays whose order carries no information, so
    the natural choice is a permutation-invariant (DeepSets) summary: `sbi` ships
    one as `PermutationInvariantEmbedding`, which plays the role of the
    hand-written `DeepSet` module of the notebook.

    Args:
        embedding (str): Either "none" (feed the raw trajectory to the flow) or
            "deepset" (permutation-invariant summary).
        embedding_output_dim (int): The dimension of the learned summary.
        hidden_features (int): Width of the hidden layers of the embedding.

    Returns:
        Optional[torch.nn.Module]: The embedding network, or None when the raw
        trajectory is used as-is.
    """
    if embedding == "none":
        return None
    if embedding != "deepset":
        raise ValueError(f"Unknown embedding '{embedding}', use 'none' or 'deepset'")

    # `trial_net` is applied to every single time delay (a scalar), its outputs
    # are summed over the jump axis, and `rho` maps the sum to the summary.
    trial_net = FCEmbedding(
        input_dim=1,
        num_hiddens=hidden_features,
        num_layers=2,
        output_dim=embedding_output_dim,
    )
    return PermutationInvariantEmbedding(
        trial_net,
        trial_net_output_dim=embedding_output_dim,
        aggregation_fn="sum",
        num_hiddens=hidden_features,
        num_layers=2,
        output_dim=embedding_output_dim,
    )


def build_density_estimator(
    model: str,
    hidden_features: int,
    num_transforms: int,
    embedding_net: Optional[torch.nn.Module],
    prior: BoxUniform,
    z_score_x: str = "structured",
) -> Any:
    """Build the conditional density estimator q(theta|x) for the NPE.

    Args:
        model (str): The flow family, e.g. "zuko_maf" (the notebook default is a
            MAF), "zuko_nsf", "maf" or "nsf".
        hidden_features (int): Width of the hidden layers of the flow.
        num_transforms (int): Number of autoregressive transforms of the flow.
        embedding_net (Optional[torch.nn.Module]): The summary network, if any.
        prior (BoxUniform): The prior, whose bounds define the transformation of
            theta to an unconstrained space.
        z_score_x (str): How to standardise the trajectories. "structured" shares
            one mean and std across the jumps, which is the natural choice here
            because every entry of a trajectory is the same physical quantity;
            "independent" standardises each jump separately, and "none" leaves
            the trajectory untouched as the notebook does.

    Returns:
        Callable: The density estimator builder consumed by `NPE`.
    """
    # The notebook rescales theta to [-1, 1] by hand before training. In `sbi`
    # the equivalent is done by the estimator itself; for the zuko/mdn backends
    # we can go one better and map the bounded theta to an unconstrained space,
    # which additionally prevents the posterior from leaking outside the prior.
    kwargs: Dict[str, Any] = {
        "model": model,
        "hidden_features": hidden_features,
        "num_transforms": num_transforms,
        "z_score_x": z_score_x,
    }
    if model.startswith(UNCONSTRAINED_OK):
        kwargs["z_score_theta"] = "transform_to_unconstrained"
        # `x_dist` names the distribution whose support bounds theta; for NPE
        # that is the prior. Without it the transform has no bounds to use.
        kwargs["x_dist"] = prior
    else:
        kwargs["z_score_theta"] = "independent"
    if embedding_net is not None:
        kwargs["embedding_net"] = embedding_net
    return posterior_nn(**kwargs)


def prepare_x(taus: np.ndarray, embedding: str) -> torch.Tensor:
    """Convert trajectories to the tensor shape expected by `sbi`.

    Args:
        taus (np.ndarray): Trajectories with shape (num_pairs, njumps).
        embedding (str): The embedding choice, see `build_embedding_net`.

    Returns:
        torch.Tensor: Shape (num_pairs, njumps) without an embedding, and
        (num_pairs, njumps, 1) with the permutation-invariant one, which expects
        an explicit trial axis.
    """
    x = torch.as_tensor(taus, dtype=torch.float32)
    if embedding == "deepset":
        x = x.unsqueeze(-1)
    return x


#############
#  TRAINING #
#############
def train_npe(
    theta: torch.Tensor,
    x: torch.Tensor,
    prior: BoxUniform,
    density_estimator: Any,
    device: str,
    training_batch_size: int,
    learning_rate: float,
    validation_fraction: float,
    stop_after_epochs: int,
    max_num_epochs: int,
) -> Tuple[NPE, Any]:
    """Train the NPE on the simulated pairs.

    The notebook writes its own loop over `NPELoss` and `GDStep`; `sbi` does the
    same thing inside `NPE.train`, including the train/validation split, the
    gradient clipping and early stopping.

    Args:
        theta (torch.Tensor): Parameters, shape (num_pairs, 2).
        x (torch.Tensor): Trajectories, see `prepare_x`.
        prior (BoxUniform): The prior over the parameters.
        density_estimator (Callable): The estimator builder.
        device (str): Torch device used for training.
        training_batch_size (int): Mini-batch size.
        learning_rate (float): Adam learning rate.
        validation_fraction (float): Fraction of the pairs held out for
            validation during training.
        stop_after_epochs (int): Early stopping patience, in epochs.
        max_num_epochs (int): Hard cap on the number of epochs.

    Returns:
        tuple(NPE, Any): The trainer and the trained density estimator.
    """
    trainer = NPE(prior=prior, density_estimator=density_estimator, device=device)
    trainer.append_simulations(theta, x)
    estimator = trainer.train(
        training_batch_size=training_batch_size,
        learning_rate=learning_rate,
        validation_fraction=validation_fraction,
        stop_after_epochs=stop_after_epochs,
        max_num_epochs=max_num_epochs,
        clip_max_norm=1.0,  # same gradient clipping as the notebook's GDStep
        show_train_summary=True,
    )
    return trainer, estimator


####################
#  QUICK EVALUATION #
####################
def evaluate_observation(
    posterior: Any,
    theta_star: torch.Tensor,
    x_star: torch.Tensor,
    num_posterior_samples: int,
    outdir: Path,
) -> Dict[str, List[float]]:
    """Sample the posterior at a held-out observation and plot it against the
    ground truth, the `sbi` counterpart of `lampe`'s corner + mark_point.

    Args:
        posterior (Any): The trained posterior.
        theta_star (torch.Tensor): The true parameters, shape (2,).
        x_star (torch.Tensor): The observed trajectory, shape (njumps,) or
            (njumps, 1) when the permutation-invariant embedding is used.
        num_posterior_samples (int): Number of posterior samples to draw.
        outdir (Path): Where the figure is written.

    Returns:
        dict: Posterior mean, standard deviation and the ground truth.
    """
    samples = posterior.sample((num_posterior_samples,), x=x_star)
    samples = samples.detach().cpu()
    # `pairplot` converts what it is given with `.numpy()`, which only works for
    # tensors that are already on the host.
    theta_star = theta_star.detach().cpu()

    limits = [[DELTA_MIN, DELTA_MAX], [OMEGA_MIN, OMEGA_MAX]]
    fig, _ = pairplot(
        samples,
        points=theta_star.reshape(1, -1),
        limits=limits,
        labels=LABELS,
        upper="contour",
        diag="kde",
        figsize=(5.0, 5.0),
    )
    fig.suptitle(r"$p_\phi(\theta \mid x^*)$")
    fig.savefig(outdir / "posterior_observation.png", dpi=150)
    plt.close(fig)

    return {
        "true": theta_star.tolist(),
        "posterior_mean": samples.mean(dim=0).tolist(),
        "posterior_std": samples.std(dim=0).tolist(),
    }


################
#  DIAGNOSTICS #
################
def run_diagnostics(
    posterior: Any,
    thetas: torch.Tensor,
    xs: torch.Tensor,
    num_posterior_samples: int,
    outdir: Path,
) -> Dict[str, Any]:
    """Check that the samples really are the posterior.

    Three complementary checks are run on held-out (theta, x) pairs, which are
    prior draws and prior predictives by construction:

    * SBC on the marginals: is each parameter's posterior too narrow or too wide
      on average? A flat rank histogram cannot be rejected.
    * Expected coverage: the same machinery applied to the joint log-probability,
      which is what the notebook computes with `expected_coverage_mc`. In the CDF
      plot, below the diagonal means over-confident.
    * TARP: a necessary and sufficient check of posterior correctness, read like
      SBC when the default (uniform) references are used.

    Args:
        posterior (Any): The trained posterior.
        thetas (torch.Tensor): Held-out parameters, shape (num_pairs, 2).
        xs (torch.Tensor): Held-out trajectories, see `prepare_x`.
        num_posterior_samples (int): Posterior samples drawn per held-out pair.
        outdir (Path): Where the figures are written.

    Returns:
        dict: The numerical summaries of the three checks.
    """
    results: Dict[str, Any] = {"num_diagnostic_pairs": int(thetas.shape[0])}

    # --- SBC on the marginals ------------------------------------------------
    # `run_sbc` samples the posterior, so `thetas` and `xs` have to be on its
    # device; its outputs come back on that device and are moved to the host
    # here, because `check_sbc` and the plots go through numpy.
    ranks, dap_samples = run_sbc(
        thetas, xs, posterior, num_posterior_samples=num_posterior_samples
    )
    ranks, dap_samples = ranks.cpu(), dap_samples.cpu()
    stats = check_sbc(
        ranks, thetas.cpu(), dap_samples, num_posterior_samples=num_posterior_samples
    )
    results["sbc"] = {k: np.asarray(v).tolist() for k, v in stats.items()}
    fig, _ = sbc_rank_plot(
        ranks,
        num_posterior_samples,
        plot_type="hist",
        num_bins=20,
        parameter_labels=LABELS,
    )
    fig.savefig(outdir / "sbc_rank_histogram.png", dpi=150)
    plt.close(fig)

    # --- Expected coverage (ranks of the joint log-probability) --------------
    cov_ranks, _ = run_sbc(
        thetas,
        xs,
        posterior,
        num_posterior_samples=num_posterior_samples,
        reduce_fns=posterior.log_prob,
    )
    cov_ranks = cov_ranks.cpu()
    fig, _ = sbc_rank_plot(
        cov_ranks,
        num_posterior_samples,
        plot_type="cdf",
        num_bins=20,
        parameter_labels=[r"$\log p_\phi(\theta \mid x)$"],
    )
    fig.savefig(outdir / "expected_coverage.png", dpi=150)
    plt.close(fig)

    # --- TARP ----------------------------------------------------------------
    ecp, alpha = run_tarp(
        thetas, xs, posterior, num_posterior_samples=num_posterior_samples
    )
    ecp, alpha = ecp.cpu(), alpha.cpu()
    atc, ks_pval = check_tarp(ecp, alpha)
    results["tarp"] = {"atc": float(atc), "ks_pval": float(ks_pval)}
    fig, _ = plot_tarp(ecp, alpha)
    fig.savefig(outdir / "tarp.png", dpi=150)
    plt.close(fig)

    return results


def posterior_predictive_check(
    posterior: Any,
    x_star: torch.Tensor,
    num_simulations: int,
    njumps: int,
    outdir: Path,
    seed: Optional[int] = None,
) -> Dict[str, Any]:
    """Push posterior samples back through the quantum trajectory simulator and
    check that the observed trajectory is not an outlier of the resulting cloud.

    This is the one check that can tell whether the simulator can produce the
    observation at all, which no calibration check sees. It needs `qutip`, so the
    import is local and the check is opt-in through `--run_ppc`.

    Args:
        posterior (Any): The trained posterior.
        x_star (torch.Tensor): The observed trajectory.
        num_simulations (int): Number of posterior draws to re-simulate. The
            simulator is slow, so this is deliberately small.
        njumps (int): Number of time delays per simulated trajectory.
        outdir (Path): Where the figure is written.
        seed (Optional[int]): Seed for the simulator.

    Returns:
        dict: The observed and predicted summary statistics.
    """
    from paramest_nn.quantum_tools import generate_clicks_TLS

    if seed is not None:
        np.random.seed(seed)

    theta_pp = posterior.sample((num_simulations,), x=x_star).detach().cpu().numpy()
    x_pp = np.stack([generate_clicks_TLS(theta, njumpsMC=njumps) for theta in theta_pp])

    taus_obs = x_star.detach().cpu().numpy().reshape(-1)
    stats_obs = np.array([taus_obs.mean(), taus_obs.std()])
    stats_pp = np.stack([x_pp.mean(axis=1), x_pp.std(axis=1)], axis=1)

    # The time delays are heavy tailed, so compare them on a logarithmic axis.
    pooled = np.concatenate([x_pp.reshape(-1), taus_obs])
    bins = np.logspace(np.log10(pooled.min()), np.log10(pooled.max()), 40)

    fig, axes = plt.subplots(1, 2, figsize=(9.5, 4.0))
    axes[0].hist(
        x_pp.reshape(-1), bins=bins, density=True, alpha=0.6, label="predictive"
    )
    axes[0].hist(taus_obs, bins=bins, density=True, histtype="step", label=r"$x^*$")
    axes[0].set(
        xlabel=r"$\tau$", ylabel="density", title="Pooled time delays", xscale="log"
    )
    axes[0].legend()

    axes[1].scatter(stats_pp[:, 0], stats_pp[:, 1], s=12, alpha=0.6, label="predictive")
    axes[1].scatter(*stats_obs, marker="*", s=180, color="red", label=r"$x^*$")
    axes[1].set(
        xlabel=r"mean $\tau$",
        ylabel=r"std $\tau$",
        title="Summary statistics",
        xscale="log",
        yscale="log",
    )
    axes[1].legend()

    fig.tight_layout()
    fig.savefig(outdir / "posterior_predictive_check.png", dpi=150)
    plt.close(fig)

    return {
        "observed_summary": stats_obs.tolist(),
        "predictive_summary_mean": stats_pp.mean(axis=0).tolist(),
        "predictive_summary_std": stats_pp.std(axis=0).tolist(),
    }


def resolve_device(device: str) -> str:
    """Turn "auto" into the best device available on this machine.

    Args:
        device (str): "auto", "cpu", "cuda" or "mps".

    Returns:
        str: The device string handed to `sbi`.
    """
    if device != "auto":
        return device
    if torch.cuda.is_available():
        return "cuda"
    if torch.backends.mps.is_available():
        return "mps"
    return "cpu"


#############
#  MAIN     #
#############
def main(
    datapath: str = "data/training-trajectories/",
    outdir: str = "data/models/npe-sbi-2D/",
    num_train: int = 512_000,
    num_diagnostic: int = 256,
    obs_index: int = 1_000_000,
    model: str = "zuko_nsf",
    embedding: str = "none",
    hidden_features: int = 128,
    num_transforms: int = 3,
    embedding_output_dim: int = 16,
    z_score_x: str = "structured",
    training_batch_size: int = 256,
    learning_rate: float = 1e-3,
    validation_fraction: float = 0.1,
    stop_after_epochs: int = 20,
    max_num_epochs: int = 50,
    num_posterior_samples: int = 2**14,
    num_diagnostic_samples: int = 1000,
    device: str = "auto",
    seed: int = 0,
    plot_data: bool = True,
    run_ppc: bool = False,
    num_ppc_simulations: int = 50,
) -> None:
    """Train and validate an amortized NPE posterior over (Delta, Omega) with
    the `sbi` package, following the steps of `notebooks/4-NPE.ipynb`.

    Args:
        datapath (str): Root folder holding the downloaded training data.
        outdir (str): Folder where figures, the posterior and the summary go.
        num_train (int): Number of (theta, x) pairs used for training.
        num_diagnostic (int): Number of held-out pairs used for SBC, expected
            coverage and TARP. A few hundred is the usual budget.
        obs_index (int): Index of the pair used as the example observation. It
            must fall outside the training and diagnostic slices.
        model (str): Flow family for q(theta|x), e.g. "zuko_maf" or "zuko_nsf".
        embedding (str): "none" or "deepset" (permutation-invariant summary).
        hidden_features (int): Width of the hidden layers of the flow.
        num_transforms (int): Number of autoregressive transforms of the flow.
        embedding_output_dim (int): Dimension of the learned trajectory summary.
        z_score_x (str): Standardisation of the trajectories, one of
            "structured", "independent" or "none".
        training_batch_size (int): Mini-batch size during training.
        learning_rate (float): Adam learning rate.
        validation_fraction (float): Fraction of pairs held out for validation.
        stop_after_epochs (int): Early stopping patience, in epochs.
        max_num_epochs (int): Hard cap on the number of training epochs.
        num_posterior_samples (int): Samples drawn at the example observation.
        num_diagnostic_samples (int): Posterior samples per diagnostic pair.
        device (str): "auto", "cpu", "cuda" or "mps".
        seed (int): Seed for torch and numpy.
        plot_data (bool): Whether to write the exploratory data figure.
        run_ppc (bool): Whether to run the posterior predictive check, which
            re-runs the `qutip` simulator and is therefore slow.
        num_ppc_simulations (int): Posterior draws re-simulated for the check.
    """
    torch.manual_seed(seed)
    np.random.seed(seed)

    out = Path(outdir)
    out.mkdir(parents=True, exist_ok=True)
    device = resolve_device(device)
    print(f"Running on device: {device}")

    # --- Read training data --------------------------------------------------
    params, taus = get_training_pairs(datapath)
    njumps = taus.shape[-1]
    print(f"We have {len(params)} trajectories of {njumps} jumps")
    print(f"Parameter ranges: min={params.min(axis=0)}, max={params.max(axis=0)}")

    if plot_data:
        plot_training_data(params, taus, out)

    # The three slices below must not overlap: training, diagnostics, and the
    # single example observation.
    assert num_train + num_diagnostic <= len(params), (
        f"num_train + num_diagnostic = {num_train + num_diagnostic} exceeds the "
        f"{len(params)} available pairs"
    )
    assert obs_index >= num_train + num_diagnostic, (
        f"obs_index={obs_index} falls inside the training or diagnostic slice; "
        f"pick an index >= {num_train + num_diagnostic}"
    )
    assert obs_index < len(params), f"obs_index={obs_index} is out of range"

    # The training pairs stay on the CPU: `train` moves them across one
    # mini-batch at a time, so keeping the whole set on the accelerator would
    # only waste its memory.
    theta_train = torch.as_tensor(params[:num_train], dtype=torch.float32)
    x_train = prepare_x(taus[:num_train], embedding)

    # Everything handed to the *trained* posterior does have to live on its
    # device: `sbi` only moves `x` for the observation set with `set_default_x`,
    # and leaves a tensor passed straight to `sample(x=...)` where it is.
    diag_slice = slice(num_train, num_train + num_diagnostic)
    theta_diag = torch.as_tensor(params[diag_slice], dtype=torch.float32).to(device)
    x_diag = prepare_x(taus[diag_slice], embedding).to(device)

    theta_star = torch.as_tensor(params[obs_index], dtype=torch.float32).to(device)
    x_star = prepare_x(taus[obs_index][None, :], embedding)[0].to(device)

    # --- Train the NPE -------------------------------------------------------
    prior = build_prior(device=device)
    embedding_net = build_embedding_net(
        embedding, embedding_output_dim, hidden_features
    )
    density_estimator = build_density_estimator(
        model, hidden_features, num_transforms, embedding_net, prior, z_score_x
    )
    trainer, estimator = train_npe(
        theta_train,
        x_train,
        prior,
        density_estimator,
        device,
        training_batch_size,
        learning_rate,
        validation_fraction,
        stop_after_epochs,
        max_num_epochs,
    )
    posterior = trainer.build_posterior(estimator)
    print(posterior)

    summary: Dict[str, Any] = {
        "config": {
            "parameters": PARAMETERS,
            "num_train": num_train,
            "num_diagnostic": num_diagnostic,
            "obs_index": obs_index,
            "model": model,
            "embedding": embedding,
            "hidden_features": hidden_features,
            "num_transforms": num_transforms,
            "embedding_output_dim": embedding_output_dim,
            "z_score_x": z_score_x,
            "training_batch_size": training_batch_size,
            "learning_rate": learning_rate,
            "max_num_epochs": max_num_epochs,
            "device": device,
            "seed": seed,
        },
        "training": {
            "epochs": int(trainer.summary["epochs_trained"][-1]),
            "best_validation_loss": float(trainer.summary["best_validation_loss"][-1]),
        },
    }

    # --- Quick evaluation on a held-out observation --------------------------
    summary["observation"] = evaluate_observation(
        posterior, theta_star, x_star, num_posterior_samples, out
    )
    print(f"Observation summary: {summary['observation']}")

    # --- Diagnostics ---------------------------------------------------------
    # A posterior that has not been checked is not a result: always run these.
    summary["diagnostics"] = run_diagnostics(
        posterior, theta_diag, x_diag, num_diagnostic_samples, out
    )
    print(f"Diagnostics: {summary['diagnostics']}")

    if run_ppc:
        summary["posterior_predictive"] = posterior_predictive_check(
            posterior, x_star, num_ppc_simulations, njumps, out, seed=seed
        )
        print(f"Posterior predictive check: {summary['posterior_predictive']}")

    # --- Save the trained posterior and the summary --------------------------
    torch.save(estimator.state_dict(), out / "npe_density_estimator.pt")
    with open(out / "npe_posterior.pkl", "wb") as f:
        pickle.dump(posterior, f)
    with open(out / "summary.json", "w") as f:
        json.dump(summary, f, indent=2)
    print(f"Wrote results to {out.resolve()}")
    return


if __name__ == "__main__":
    fire.Fire(main)
