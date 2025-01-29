# import OS module
import argparse
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

    # -------------------- to plot image trials --------------------
    # analyzer = EyeTrackingDataImage(user, session, trials, show)
    # trials = [4.1, 4.3]
    # trials = [4.1, 4.3, 6.1, 6.3, 7.1, 7.3, 8.1, 8.2, 8888.1, 8888.2, 9999.1, 9999.2]
    # for trial in trials:
    # analyzer.plot_image_trial(trial, fixations=True, coordinates=True, calibrate=False)
    # analyzer.plot_image_trial(trial, fixations=True, coordinates=True, calibrate=True)
    # ---------------------------------------------------------------

    # -------------------- asign fixations to words,  print them on the image--------------------
    users = {20: 2}
    sessions = [1]
    # EyeTrackingAnalyser.compute_images_and_show(
    #     users,
    #     sessions,
    #     path=path,
    #     plot_fixations_words=False,
    #     calibrate=True,
    #     print_fix_distance=False,
    # )
    # EyeTrackingAnalyser.compute_images_and_show(
    #     users, sessions, path=path, plot_fixations_words=False, calibrate=True, print_fix_distance=True,  filter_value=0, filter_proportion=0.1
    # )
    # EyeTrackingAnalyser.compute_images_and_show(
    #     users, sessions, path=path, plot_fixations_words=False, calibrate=True, print_fix_distance=True,  filter_value=0, filter_proportion=0.1
    # )
    # EyeTrackingAnalyser.compute_images_and_show_removed(
    #     users, sessions, path=path, plot_fixations_words=False, calibrate=True, print_fix_distance=True, filter_value=0, filter_proportion=0.1
    # )

    # ----------------------------------------------------------------

    # ------------------------------compute_fixations_and_save----------------------------------
    users = {
        1: 1,
        # 2: 2,
        3: 3,
        4: 4,
        5: 5,
        6: 6,
        7: 7,
        8: 8,
        9: 5,
        10: 5,
        # 11: 6,
        12: 1,
        13: 1,
        14: 6,
        15: 8,
        16: 8,
        17: 3,
        18: 3,
        19: 2,
        20: 2,
        21: 4,
        22: 4,
        23: 7,
        24: 7,
        25: 2,
        26: 6,
    }
    # users = {1: 1, 2: 2}
    sessions = [1]
    path = str(pathlib.Path(__file__).parent.resolve().parent.resolve()) + "/oasstetc/raw_data"

    EyeTrackingAnalyser.compute_fixations_process_and_save(
        users=users, sessions=sessions, path=path
    )
    # ----------------------------------------------------------------

   
