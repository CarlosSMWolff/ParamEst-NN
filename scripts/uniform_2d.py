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
from paramest_nn.quantum_tools import create_directory

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
# number of trajectories to use for each parameter set (minimum:1, maximum:10000)
num_trajs = 1
# TLS parameters
gamma = 1.0
# Trajectories info
njumps = 48
# path to downloaded data and parameters

datapath = 'data/'
path_param = datapath +"validation-trajectories/2D-delta-omega/validation-deltas-2D-delta-omega-nsets-10000.npy"
path_tau = datapath + "validation-trajectories/2D-delta-omega/validation-trajectories-2D-delta-omega-nsets-10000.npy"


#############
#  GET DATA #
#############
def get_traj_and_delta(
    filenametraj: str, filenamedelta: str
) -> Tuple[np.array, np.array]:
    """Get the arrays containing all trajectories and all
    parameters (delta,omega) values for validation.

    Args:
        filenametraj (str): The file name for the trajectories
        filenamedelta (str): The filename for the deltas

    Returns:
        tuple(np.array, np.array): A tuple of 2 arrays (traj, delta);
        the trajectories first and the deltas second
    """
    ft = Path(filenametraj)
    fd = Path(filenamedelta)
    assert ft.is_file(), f"File not found {filenametraj}"
    assert fd.is_file(), f"File not found {filenamedelta}"
    return (np.load(ft), np.load(fd))


######################
#  DEFINE LIKELIHOOD #
######################
def single_click_probability(params: np.ndarray, tau: float) -> float:
    """The probability to observe a click after some time tau
       for a 2-level system

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
    """Likelihood function that will tell us how probable the data is
       given our model with parameters delta and omega.

    Args:
        params (np.ndarray): The parameter of the system.
        In our case this is a 2D array representing the detuning and frequency
        data (np.ndarray): The trajectory data of the time delays

    Returns:
        np.ndarray: The likelihood function for each observation
        (we compute it after each jump (click))
    """
    prob_list = np.asarray([single_click_probability(params, tau) for tau in data])
    log_likelihood = np.sum(np.log(prob_list)) + 1e-300

    return log_likelihood


##################
# UNIFORM PRIOR  #
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
#  MAIN     #
#############
def main(params_id: int) -> None:
    """Main function running inference with UltraNest on a 2D parameter space
    with data selected from validation trajectories
    for a fixed pair of parameters (delta,omega)

    Args:
        params_id (int): The index corresponding to a pair of parameters (delta,omega)
        from a grid. The full grid is saved on a file.
    """
    # get data from downloaded files
    trajs, params = get_traj_and_delta(path_tau, path_param)
    # extract trajectories for validation
    param = params[params_id]
    validation_trajs = trajs[params_id][:num_trajs]
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
        # Save posterior and ML to dataframe
        df = pd.DataFrame(results["posterior"] | results["maximum_likelihood"])
        # add columns for the trajectory number and the name of parameters
        df["traj_id"] = traj_id
        df["params"] = parameters
        post = pd.concat([post, df], ignore_index=True)
    # add delta and omega as columns in the dataframe
    post["Delta"] = param[0]
    post["Omega"] = param[1]
    # remove non-informative columns
    post.drop(
        columns=["logl", "information_gain_bits", "point_untransformed"], inplace=True
    )
    # name of the results file
    results_folder = "data/results_cache/bayesian_estimation/2D/estimationBayes2DUltranest/test/"
    create_directory(results_folder)
    results_name = results_folder +f"uniform_2d_{str(params_id).zfill(4)}.csv"
    # save to file
    post.to_csv(results_name, index=False)
    return


if __name__ == "__main__":
    fire.Fire(main)
