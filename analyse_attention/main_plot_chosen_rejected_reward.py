from models.plotter import Plotter
import pandas as pd

from models.compare_att import CompareAttention
import pathlib
# Define the path and files (same as your code)
path = str(pathlib.Path(__file__).parent.resolve().parent.resolve())
path_normal = path + "/oasstetc_data/attention/results/"
path_reward = path + "/oasstetc_data/attention_reward/results/"
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
dfs, dfs_reward = {}, {}
for file, gaze_signal in files.items():
    dfs[gaze_signal] = {}
    dfs[gaze_signal]["chosen"] = pd.read_csv(
        path_normal + "chosen/" + file, sep=";", index_col=0
    )
    dfs[gaze_signal]["rejected"] = pd.read_csv(
        path_normal + "rejected/" + file, sep=";", index_col=0
    )
    dfs[gaze_signal]["chosen"] = dfs[gaze_signal]["chosen"].dropna()
    dfs[gaze_signal]["rejected"] = dfs[gaze_signal]["rejected"].dropna()
    dfs[gaze_signal]["chosen_all"] = pd.read_csv(
        path_normal
        + "chosen/"
        + file.replace("correlation_trials_", "correlation_trials_alldata"),
        sep=";",
        index_col=0,
    )
    dfs[gaze_signal]["rejected_all"] = pd.read_csv(
        path_normal
        + "rejected/"
        + file.replace("correlation_trials_", "correlation_trials_alldata"),
        sep=";",
        index_col=0,
    )


for file, gaze_signal in files.items():
    dfs_reward[gaze_signal] = {}
    dfs_reward[gaze_signal]["chosen"] = pd.read_csv(
        path_reward + "chosen/" + file, sep=";", index_col=0
    )
    dfs_reward[gaze_signal]["rejected"] = pd.read_csv(
        path_reward + "rejected/" + file, sep=";", index_col=0
    )
    dfs_reward[gaze_signal]["chosen"] = dfs_reward[gaze_signal]["chosen"].dropna()
    dfs_reward[gaze_signal]["rejected"] = dfs_reward[gaze_signal]["rejected"].dropna()
    dfs_reward[gaze_signal]["chosen_all"] = pd.read_csv(
        path_reward
        + "chosen/"
        + file.replace("correlation_trials_", "correlation_trials_alldata"),
        sep=";",
        index_col=0,
    )
    dfs_reward[gaze_signal]["rejected_all"] = pd.read_csv(
        path_reward
        + "rejected/"
        + file.replace("correlation_trials_", "correlation_trials_alldata"),
        sep=";",
        index_col=0,
    )


for gaze_signal, df1 in dfs.items():
    df2 = dfs_reward[gaze_signal]
    p_values_df1 = CompareAttention.compute_posthoc_comparisons_correlation(
        df1["chosen_all"], df1["rejected_all"]
    )
    p_values_df2 = CompareAttention.compute_posthoc_comparisons_correlation(
        df2["chosen_all"], df2["rejected_all"]
    )
    Plotter.plot_gaze_signal_chosenrejected_reward(
        path_reward,
        df1=df1,
        df2=df2,
        gaze_signal=gaze_signal,
        tag=level,
        plot_std=False,
        p_values_df1=p_values_df1,
        p_values_df2=p_values_df2,
    )
