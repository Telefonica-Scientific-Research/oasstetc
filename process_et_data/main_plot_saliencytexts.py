import sys
import pathlib

sys.path.append("../..")
path = str(pathlib.Path(__file__).parent.resolve())
sys.path.append(path)
path = str(pathlib.Path(__file__).parent.resolve().parent.resolve())
sys.path.append(path)
path = str(pathlib.Path(__file__).parent.resolve().parent.resolve().parent.resolve())
sys.path.append(path)
import numpy as np
from eyetrackpy.data_printer.models.fixations_plotter import FixationsPlotter
import re
import pandas as pd


def load_gaze_features(folder, pattern=r"word_cor_image_fixations_(\d+\.\d+)"):
    files = search_files(folder, pattern=pattern)
    fixations_trials = {}
    for trial, file in files.items():
        fixations_trials[trial] = read_data_trial(file)
    return fixations_trials

def read_data_trial(file):
    data_trial = pd.read_csv(
        file,
        sep=";",
        index_col=0,
    )
    data_trial = data_trial.loc[:, ~data_trial.columns.str.contains("^Unnamed")]
    return data_trial


def search_files(folder, pattern=r"word_cor_image_fixations_(\d+\.\d+)"):
    if isinstance(folder, str):
        folder = pathlib.Path(folder)
    file_paths = list(folder.rglob("*"))
    files = {}

    for file in file_paths:
        # The regex pattern
        # Perform the match
        match = re.search(pattern, str(file))
        # Check if there is a match and extract the number
        if match:
            trial = match.group(1)
            files[trial] = file

    return files


def plot_fixations_real(
    trial,
    data,
    features_to_plot: list,
    path: str,
    auxiliar_name: str = "",
    max_char_per_line: int = 50,
):
    # ----------------- Plot fixations ----------------

    save_path_prev = path + "plots_saliency/"
    if not pathlib.Path(save_path_prev).exists():
        pathlib.Path(save_path_prev).mkdir(parents=True, exist_ok=True)
    for feature in features_to_plot:
        save_path = (
            save_path_prev
            + "fix_"
            + str(trial)
            + "_"
            + str(feature)
            + "_"
            + str(auxiliar_name)
            + "_plot.png"
        )
        try:
            words_with_values = list(zip(data["text"], data[feature]))
        except Exception as e:
            print(e)
        FixationsPlotter.plot_fixations(
            words_with_values, max_char_per_line=max_char_per_line, save_path=save_path
        )


def plot_fixations_synthetic(
    trial,
    data,
    path: str,
    auxiliar_name="",
    features_to_plot: list = ["nFix", "FFD", "GPT", "TRT", "fixProp"],
    max_char_per_line: int = 50,
):
    # ----------------- Plot fixations ----------------
    save_path_prev = path + "plots_saliency/"
    if not pathlib.Path(save_path_prev).exists():
        pathlib.Path(save_path_prev).mkdir(parents=True, exist_ok=True)
    for feature in features_to_plot:
        save_path = (
            save_path_prev
            + "fix_"
            + str(trial)
            + "_"
            + str(feature)
            + "_"
            + str(auxiliar_name)
            + "_plot.png"
        )
        words_with_values = list(zip(data["text"], data[feature]))
        FixationsPlotter.plot_fixations(
            words_with_values, max_char_per_line=max_char_per_line, save_path=save_path
        )


def print_fixations_userset(user_set: int, path: str):
    max_char_per_line = 40
    folder_gaze_features_real = (
        path + "gaze_features_real" + "/set_" + str(user_set) + "/"
    )
    # load fixations of user and concat
    real_gaze_fixations_user = load_gaze_features(
        folder=folder_gaze_features_real, pattern=r"word_cor_image_fixations_(\d+\.\d+)"
    )
    # features_to_plot = ["fix_duration", "first_fix_duration", "fix_number"]
    features_to_plot = ["fix_duration"]

    for trial, data_trial in real_gaze_fixations_user.items():
        if str(trial).endswith(".1"):
            plot_fixations_real(
                trial=trial,
                data=data_trial,
                path=folder_gaze_features_real,
                features_to_plot=features_to_plot,
                auxiliar_name="chosen",
                max_char_per_line=max_char_per_line,
            )
        else:
            plot_fixations_real(
                trial=trial,
                data=data_trial,
                path=folder_gaze_features_real,
                auxiliar_name="rejected",
                features_to_plot=features_to_plot,
                max_char_per_line=max_char_per_line,
            )

    # features_to_plot = ["nFix", "FFD", "TRT"]
    # features_to_plot = ["fix_duration", "first_´fix_duration", "fix_number"]
    features_to_plot = ["fix_duration"]
    folder_gaze_features_syn = (
        path + "gaze_features_synthetic" + "/set_" + str(user_set) + "/"
    )
    synthetic_gaze_fixations_user = load_gaze_features(
        folder=folder_gaze_features_syn, pattern=r"word_gaze_synthetic_(\d+\.\d+)"
    )
    for trial, data_trial in synthetic_gaze_fixations_user.items():
        if str(trial).endswith(".1"):
            plot_fixations_synthetic(
                trial=trial,
                data=data_trial,
                path=folder_gaze_features_syn,
                features_to_plot=features_to_plot,
                auxiliar_name="chosen",
                max_char_per_line=max_char_per_line,
            )
        else:
            plot_fixations_synthetic(
                trial=trial,
                data=data_trial,
                path=folder_gaze_features_syn,
                auxiliar_name="rejected",
                features_to_plot=features_to_plot,
                max_char_per_line=max_char_per_line,
            )


if __name__ == "__main__":
    print_fixations_userset(path="oasstetc/", user_set=2)
