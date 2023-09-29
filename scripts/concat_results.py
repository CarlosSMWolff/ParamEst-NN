from glob import glob
import pandas as pd
import numpy as np

# batch number and final output file for predictions
idx_data = 0
file_output = f"data/results_cache/bayesian_estimation/2D/estimationBayes2DUltranest/uniform_2d_predictions_{idx_data}.npy"
# these are the files, one for each parameter pair, with the posterior info
file_pattern = "uniform_2d_*.csv"
files = sorted(glob(file_pattern))
# load posteriors for each parameter file
df = pd.concat([pd.read_csv(f) for f in files], ignore_index=True)
# keep only the mean of the posterior
data = df[["mean", "params"]]
omega_preds_mean = np.array(data[data.params == "omega"].iloc[:, 0])
delta_preds_mean = np.array(data[data.params == "delta"].iloc[:, 0])
# write predictions to output file
np.save(file_output, np.asarray([omega_preds_mean, delta_preds_mean]))
