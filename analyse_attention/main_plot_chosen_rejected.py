import pandas as pd


from models.plotter import Plotter

from models.compare_att import CompareAttention
import pathlib
# Define the path and files (same as your code)

path = str(pathlib.Path(__file__).parent.resolve().parent.resolve()) + "/oasstetc_data/attention/results/"
level = "trials"
files = {
    # "completed/correlation_" + level + "_fix_duration.csv": "TRT_f",
    "completed/correlation_" + level + "_fix_duration_n.csv": "TRT_n_f",
    # "completed/correlation_" + level + "_first_fix_duration.csv": "FFD_f",
    "completed/correlation_" + level + "_first_fix_duration_n.csv": "FFD_n_f",
    "completed/correlation_" + level + "_fix_number.csv": "nFix_f",
    "not_filtered/correlation_" + level + "_fix_duration_n.csv": "TRT_n_not_f",
    # "not_filtered/correlation_" + level + "_fix_duration.csv": "TRT_not_f",
    # "not_filtered/correlation_" + level + "_first_fix_duration.csv": "FFD_not_f",
    "not_filtered/correlation_" + level + "_first_fix_duration_n.csv": "FFD_n_not_f",
    "not_filtered/correlation_" + level + "_fix_number.csv": "nFix_not_f",
}


# Load the data
dfs = {}
for file, gaze_signal in files.items():
    dfs[gaze_signal] = {}
    dfs[gaze_signal]["chosen"] = pd.read_csv(
        path + "chosen/" + file, sep=";", index_col=0
    )
    dfs[gaze_signal]["rejected"] = pd.read_csv(
        path + "rejected/" + file, sep=";", index_col=0
    )
    dfs[gaze_signal]["chosen_all"] = pd.read_csv(
        path
        + "chosen/"
        + file.replace("correlation_trials_", "correlation_trials_alldata"),
        sep=";",
        index_col=0,
    )
    dfs[gaze_signal]["rejected_all"] = pd.read_csv(
        path
        + "rejected/"
        + file.replace("correlation_trials_", "correlation_trials_alldata"),
        sep=";",
        index_col=0,
    )
    dfs[gaze_signal]["chosen"] = dfs[gaze_signal]["chosen"].dropna()
    dfs[gaze_signal]["rejected"] = dfs[gaze_signal]["rejected"].dropna()


for gaze_signal, df in dfs.items():
    p_values = CompareAttention.compute_posthoc_comparisons_correlation(
        dfs[gaze_signal]["chosen_all"], dfs[gaze_signal]["rejected_all"]
    )
    Plotter.plot_gaze_signal_chosenrejected(
        path, df, gaze_signal, tag=level, plot_std=False, p_values=p_values
    )
