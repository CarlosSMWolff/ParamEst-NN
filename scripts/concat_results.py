from glob import glob
import pandas as pd

file_pattern = "uniform_2d_*.csv"
file_output = "uniform_2d_all.csv"
files = sorted(glob(file_pattern))
df = pd.concat([pd.read_csv(f) for f in files], ignore_index=True)
df.to_csv(file_output, index=False)
