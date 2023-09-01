"""
Script to generate the posterior estimates for trajectories saved on file 
generated with a fixed delta
Based on ultranest and with a 2D parameter space
The prior is uniform
"""
from typing import Tuple
import numpy as np
import pandas as pd
from tqdm import tqdm
from pathlib import Path
from ultranest import ReactiveNestedSampler
from functools import partial
import logging
import fire

logger = logging.getLogger("ultranest")
logger.addHandler(logging.NullHandler())
logger.setLevel(logging.WARNING)

parameters = ["delta", "omega"]
# Parameter ranges: detuning delta and frequency omega
# the detuning delta can only take values between 0 and delta_max
delta_min = 0.0
delta_max = 3.0
# the frequency of the system can only take values between 0.25 and 5
omega_min = 0.25
omega_max = 5.0
# number of trajectories to use for each parameter set
num_trajs = 1
## TLS parameters
gamma = 1.0
# Trajectories info
njumps = 48
# path
trajs_name = "data/training-trajectories/2D-delta-omega/taus-2D.npy"
params_name = (
    "data/training-trajectories/2D-delta-omega/param_rand_list-2D.npy"
)


#############
## GET DATA #
#############
def to_time_delay_matrix(data: np.array) -> np.array:
    """This function takes a 2D array with the times of the jumps for many trajectories and transforms it into a 2D array of time delays between jumps for each trajectory

    Args:
        data (np.array): The 2D array with the times of each jump

    Returns:
        np.array: The 2D array with the time delays between jumps
    """
    assert len(data.shape) == 2, "Must be a 2D array"
    assert data.shape[1] == njumps, f"Number of jumps does not correspond to {njumps}"
    return np.concatenate(
        (np.reshape(data[:, 0], (data.shape[0], 1)), np.diff(data)), axis=1
    )


def get_traj_and_delta(
    filenametraj: str, filenamedelta: str
) -> Tuple[np.array, np.array]:
    """Get the arrays containing all trajectories and all delta values for validation.

    Args:
        filenametraj (str): The file name for the trajectories
        filenamedelta (str): The filename for the deltas

    Returns:
        tuple(np.array, np.array): A tuple of 2 arrays (traj, delta); the trajectories first and the deltas second
    """
    ft = Path(filenametraj)
    fd = Path(filenamedelta)
    assert ft.is_file(), f"File not found {filenametraj}"
    assert fd.is_file(), f"File not found {filenamedelta}"
    return (np.load(ft), np.load(fd))


######################
## DEFINE LIKELIHOOD #
######################
def single_click_probability(params: np.ndarray, tau: float) -> float:
    """The probability to observe a click after some time tau for a 2-level system

    Args:
        params (np.ndarray): The array of parameters for the Hamiltonian (2D)
        tau (float): The delay between on click and the next.

    Returns:
        float: The probability of a jump (click) with this time delay
    """
    (delta, omega) = params
    return (
        8
        * np.exp(-tau / 2)
        * omega**2
        / np.sqrt(-64 * omega**2 + (1 + 4 * delta**2 + 16 * omega**2) ** 2)
        * (
            -np.cos(
                tau
                * np.sqrt(
                    -1
                    + 4 * delta**2
                    + 16 * omega**2
                    + np.sqrt(
                        -64 * omega**2 + (1 + 4 * delta**2 + 16 * omega**2) ** 2
                    )
                )
                / (2 * np.sqrt(2))
            )
            + np.cosh(
                tau
                * np.sqrt(
                    1
                    - 4 * delta**2
                    - 16 * omega**2
                    + np.sqrt(
                        -64 * omega**2 + (1 + 4 * delta**2 + 16 * omega**2) ** 2
                    )
                )
                / (2 * np.sqrt(2))
            )
        )
    )


def log_likelihood(params: np.ndarray, data: np.ndarray) -> np.ndarray:
    """Likelihood function that will tell us how probable the data is given our model with parameters delta and omega.

    Args:
        params (np.ndarray): The parameter of the system. In our case this is a 2D array representing the detuning and frequency
        data (np.ndarray): The trajectory data of the time delays

    Returns:
        np.ndarray: The likelihood function for each observation (we compute it after each jump)
    """
    prob_list = np.asarray([single_click_probability(params, tau) for tau in data])
    log_likelihood = np.sum(np.log(prob_list)) + 1e-300

    return log_likelihood


##################
## PRIOR UNIFORM #
##################
def prior_transform_uniform(cube):
    # the argument, cube, consists of values from 0 to 1
    # we have to convert them to physical scales

    params = cube.copy()
    # let delta go from delta_min to delta_max
    params[0] = cube[0] * (delta_max - delta_min) + delta_min
    # let omega go from omega_min to omega_max
    params[1] = cube[1] * (omega_max - omega_min) + omega_min
    # the parameters have independent priors
    return params


#############
## MAIN     #
#############
def main(params_id: int):
    results_name = f"uniform_2d_{params_id}.csv"
    trajs, params = get_traj_and_delta(trajs_name, params_name)
    # extract trajectories for validation
    param = params[params_id]
    validation_trajs = to_time_delay_matrix(trajs[params_id])[:num_trajs]
    print(f"Running for (delta,omega)={param}")
    print(f"Doing validation over a set of {num_trajs} trajectories of {njumps} jumps.")
    post = pd.DataFrame()
    # run ultranest for each trajectory
    for traj_id, traj in enumerate(tqdm(validation_trajs, desc="Traj loop")):
        log_likelihood_traj = partial(log_likelihood, data=traj)
        sampler = ReactiveNestedSampler(
            parameters, log_likelihood_traj, prior_transform_uniform
        )
        results = sampler.run(
            min_num_live_points=400,
            viz_callback=False,
            show_status=False,
        )
        # sampler.print_results()  # print to standard output
        df = pd.DataFrame(results["posterior"] | results["maximum_likelihood"])
        df["traj_id"] = traj_id
        df["params"] = parameters
        post = pd.concat([post, df], ignore_index=True)
    # add delta as a column in the dataframe
    post["Delta"] = param[0]
    post["Omega"] = param[1]
    # remove non-informative columns
    post.drop(
        columns=["logl", "information_gain_bits", "point_untransformed"],
        inplace=True
    )
    # save to file
    post.to_csv(results_name, index=False)


if __name__ == "__main__":
    fire.Fire(main)
