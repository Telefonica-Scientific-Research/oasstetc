from models.plotter import Plotter
import pandas as pd


# Define the path and files (same as your code)
path = "oasstetc_data/attention/results/"
level = "trials"
files = {
    "completed/correlation_" + level + "_fix_duration.csv": "TRT_f",
    "completed/correlation_" + level + "_fix_duration_n.csv": "TRT_n_f",
    "completed/correlation_" + level + "_first_fix_duration.csv": "FFD_f",
    "completed/correlation_" + level + "_first_fix_duration_n.csv": "FFD_n_f",
    "completed/correlation_" + level + "_fix_number.csv": "nFix_f",
    "not_filtered/correlation_" + level + "_fix_duration_n.csv": "TRT_n_not_f",
    "not_filtered/correlation_" + level + "_fix_duration.csv": "TRT_not_f",
    "not_filtered/correlation_" + level + "_first_fix_duration.csv": "FFD_not_f",
    "not_filtered/correlation_" + level + "_first_fix_duration_n.csv": "FFD_n_not_f",
    "not_filtered/correlation_" + level + "_fix_number.csv": "nFix_not_f",
}

# Load the data
dfs = {}
for file, gaze_signal in files.items():
    dfs[gaze_signal] = pd.read_csv(path + file, sep=";", index_col=0)
    # Drop rows full of NaNs
    dfs[gaze_signal] = dfs[gaze_signal].dropna()

for gaze_signal, df in dfs.items():
    Plotter.plot_gaze_signal(path, df, gaze_signal, tag=level, plot_std=False)
