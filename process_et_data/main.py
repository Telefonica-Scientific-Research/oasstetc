# import OS module
import argparse

# Get the list of all files and directories
import pathlib
import sys

sys.path.append("../..")
path = str(
    pathlib.Path(__file__)
    .parent.resolve()
    .parent.resolve()
    .parent.resolve()
    .parent.resolve()
    .parent.resolve()
)
sys.path.append(path)
path = str(
    pathlib.Path(__file__)
    .parent.resolve()
    .parent.resolve()
    .parent.resolve()
    .parent.resolve()
)
sys.path.append(path)
path = str(pathlib.Path(__file__).parent.resolve().parent.resolve().parent.resolve())
sys.path.append(path)
path = str(pathlib.Path(__file__).parent.resolve().parent.resolve())
sys.path.append(path)
path = str(pathlib.Path(__file__).parent.resolve())
sys.path.append(path)

# from models.eye_tracking_analyzer import EyeTrackingAnalyser
from model.eye_tracking_analyzer import EyeTrackingAnalyser


import os
import pandas as pd
import pathlib


os.environ["KMP_DUPLICATE_LIB_OK"] = "True"
if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--user", default=1, help="user")
    parser.add_argument("--session", default=1, help="session")
    parser.add_argument("--trial", default=None)

    args = parser.parse_args()
    user = args.user
    session = args.session
    trial = args.trial
    show = True
    # -------------------- read data from text and join with eye tracking tokens --------------------
    # trials = [4.1, 4.3, 6.1, 6.3, 7.1, 7.3, 8.1, 8.2, 8888.1, 8888.2, 9999.1, 9999.2]
    # analyzer = EyeTrackingDataText(user, session, trial, show)
    # analyzer.join_words_vertices()
    # ---------------------------------------------------------------

    # -------------------- compute general fetures, save then on a CSV--------------------
    # users = [1]
    # sessions = [2]
    # EyeTrackingAnalyser.compute_general_features(users, sessions)

    # ---------------------------------------------------------------

    # ---------------------------craete gaze_feature_real-------------------------------------
    # it average the reading measures per uset set (1-8)
    path = str(pathlib.Path(__file__).parent.resolve().parent.resolve()) + "/oasstetc_data/raw_data"
    path_save = str(pathlib.Path(__file__).parent.resolve().parent.resolve()) + "/oasstetc_data/gaze_features_real/"
    EyeTrackingAnalyser().average_gaze_features_real_participants(path, path_save)
    # ----------------------------------------------------------------

    # --------------------------concatenate texts and save them--------------------------------------
    # EyeTrackingAnalyser().compute_texts()
    # ----------------------------------------------------------------

    # --------------------------concatenate texts and save them--------------------------------------
    # EyeTrackingAnalyser().compute_trial_prompts()
    # ----------------------------------------------------------------

    # -----------------------------create general infor files-----------------------------------
    # path = "oasstetc_data/raw_data/"
    # EyeTrackingAnalyser().create_info_files(folder_path= path)
    # ----------------------------------------------------------------

    # -----------------------------compute number instances complete and not complete per user set-----------------------------------
    # EyeTrackingAnalyser().compute_n_instances()
    # ----------------------------------------------------------------


