import sys
import pathlib
import os
import pandas as pd
import json
import seaborn as sns
import pathlib
import numpy as np
from scipy.stats import entropy
import matplotlib.pyplot as plt
from scipy.stats import rankdata
import statsmodels.api as sm
from statsmodels.formula.api import mixedlm
from statsmodels.stats.multitest import multipletests
from collections import Counter
from scipy.stats import ttest_rel

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

from eyetrackpy.data_processor.models.eye_tracking_data_image import (
    EyeTrackingDataImage,
)
from eyetrackpy.data_processor.models.eye_tracking_data_simple import (
    EyeTrackingDataUserSet,
)
from tokenizeraligner.models.tokenizer_aligner import TokenizerAligner


def check_same_shape(dfs_dict):
    # Get the shape of the first DataFrame
    first_shape = list(dfs_dict.values())[0].shape
    # Compare the shape of all DataFrames to the first one
    return all(df.shape == first_shape for df in dfs_dict.values())


def save_to_json(data, filename="data.json"):
    # Save to JSON file
    with open(filename, "w") as f:
        json.dump(data, f)


class EyeTrackingAnalyser:
    @staticmethod
    def compute_fuzzy_entropy(data, m, r):
        def fuzzy_membership(data, r):
            # Normalize data
            normalized_data = (data - np.min(data)) / (np.max(data) - np.min(data))
            # Create membership function (example: triangular)
            membership = np.maximum(
                0, 1 - np.abs(normalized_data[:, None] - normalized_data) / r
            )
            return membership

        membership = fuzzy_membership(data, r)
        # Calculate fuzzy entropy
        fuzzy_prob = np.mean(membership, axis=0)
        entropy = -np.log(np.mean(fuzzy_prob)) if np.mean(fuzzy_prob) > 0 else 0
        return entropy

    @staticmethod
    def permutation_entropy_2d(fixations, m, delay):
        """
        Calculate the permutation entropy for a 2D time series (x, y) coordinates.

        Parameters:
        - fixations: np.array, shape (N, 2), array of (x, y) fixation coordinates.
        - m: int, embedding dimension (window size).
        - delay: int, time delay between points in each pattern.

        Returns:
        - entropy_value: float, the permutation entropy of the 2D fixation sequence.
        """
        # Generate all possible patterns of order m
        # Initialize pattern dictionary for counting
        pattern_dict = {}

        # Extract ordinal patterns for each (x, y) coordinate pair
        for i in range(len(fixations) - delay * (m - 1)):
            # Take m points with spacing defined by delay for both x and y
            window_x = fixations[i : (i + delay * m) : delay, 0]
            window_y = fixations[i : (i + delay * m) : delay, 1]

            # Get ranks for x and y separately
            ranks_x = np.argsort(window_x)  # Ranks for x
            ranks_y = np.argsort(window_y)  # Ranks for y

            # Create a combined pattern by combining ranks into a tuple
            combined_pattern = tuple((ranks_x[j], ranks_y[j]) for j in range(m))

            # Update pattern counts
            if combined_pattern in pattern_dict:
                pattern_dict[combined_pattern] += 1
            else:
                pattern_dict[combined_pattern] = 1

        # Convert counts to probabilities
        pattern_counts = np.array(list(pattern_dict.values()))
        probabilities = pattern_counts[pattern_counts > 0] / pattern_counts.sum()

        # Compute permutation entropy
        entropy_value = entropy(probabilities, base=2)

        return entropy_value

    @staticmethod
    def permutation_entropy(time_series, order=3):
        # Step 1: Create permutations
        n = len(time_series)
        # Generate all permutations of indices
        permutations = [
            np.argsort(time_series[i : i + order]) for i in range(n - order + 1)
        ]

        # Step 2: Count unique permutations
        unique_permutations, counts = np.unique(
            permutations, axis=0, return_counts=True
        )

        # Step 3: Calculate probabilities
        probabilities = counts / counts.sum()

        # Step 4: Calculate permutation entropy
        pe = -np.sum(probabilities * np.log(probabilities))

        return pe

    @staticmethod
    def compute_texts(path="oasstetc/data/"):
        datauserset = EyeTrackingDataUserSet()
        for user_set in range(1, 9):
            folder = path + "set_" + str(user_set) + "/"
            # files = datauserset.search_images_files(folder)
            # for trial, file in files.items():
            #     coordinates_data = datauserset._read_image_file(file)
            #     file_name = "word_cor_image_" + str(trial)
            #     datauserset.save_coordinates_trial(coordinates_data, file_name, folder)

            files = datauserset.search_word_coor_files(folder)
            text_trial = {}
            for trial, file in files.items():
                coordinates_data = datauserset._read_coor_trial(file)
                list_words_first = list(coordinates_data.text)
                try:
                    text = " ".join(list_words_first)
                except:
                    text = " ".join([str(x) for x in list_words_first])
                text_trial[trial] = text
            text_trial = pd.DataFrame(
                list(text_trial.items()), columns=["trial", "text"]
            )
            text_trial.to_csv(folder + "text_trials.csv", sep=";", index=False)
        return True

    @staticmethod
    def compute_trial_prompts(path="oasstetc/raw_data/"):
        texts_prompts = pd.DataFrame()
        for user_set in list(range(1, 9)):
            user = user_set
            if user_set == 2:
                user = 19
            folder_name = (
                str(path)
                + "participant_"
                + str(user)
                + "_"
                + str(user_set)
                + "/session_"
                + str(1)
            )

            for file in os.listdir(folder_name):
                # check if file is an excel
                if "TEST_eye_tracker" in file and "FINAL" in file:
                    if ".xlsm" in file:
                        # read excel
                        data = pd.read_excel(folder_name + "/" + file)
                        # data = pd.read_csv(folder_name + '/' + file, delimiter=';')
                        break
                    if ".csv" in file:
                        # read csv
                        data = pd.read_csv(folder_name + "/" + file, delimiter=";;;")
                        # data = pd.read_csv(folder_name + '/' + file, delimiter=';')
                        break
            data = data[["prompt_text", "n_resp"]]
            data = data.dropna(subset=["n_resp"])
            texts_prompts = pd.concat([data, texts_prompts])

        texts_prompts.to_csv(path + "text_prompts.csv", sep=";", index=False)
        return True

    @staticmethod
    def create_info_files(folder_path):
        def find_files_with_keyword(directory, keyword="info"):
            """Recursively search for all files with the specified keyword in their name."""
            file_paths = list(directory.rglob("*"))
            json_files = []
            for file in file_paths:
                if keyword in str(file) and str(file).endswith(str(keyword) + ".csv"):
                    # if str(file).endswith('.json'):
                    json_files.append(str(file))
            return json_files

        def create_info_file(path, keyword="info"):
            info = pd.DataFrame()
            for file in find_files_with_keyword(pathlib.Path(path), keyword):
                info_user = pd.read_csv(file, sep=";", index_col=0)
                info_user["session"] = file.split("/")[-2].split("_")[-1]
                info_user["user"] = file.split("/")[-3].split("_")[-2]
                info_user["user_set"] = file.split("/")[-3].split("_")[-1]
                info = pd.concat([info, info_user])
            return info

        def create_info_ordered(info):
            users_set = info["user_set"].unique()
            info_ordered = []
            for user_set in users_set:
                data = info[info["user_set"] == user_set]
                users = data["user"].unique()
                trials = data["trial"].unique()
                for trial in trials:
                    info_ordered_trial = {"user_set": user_set, "trial": trial}
                    for user in users:
                        trial_user = data[
                            (data["user"] == user) & (data["trial"] == trial)
                        ]
                        if trial_user.empty:
                            info_ordered_trial[user] = False
                        else:
                            info_ordered_trial[user] = True

                    info_ordered.append(info_ordered_trial)
            info_ordered = pd.DataFrame(info_ordered)
            return info_ordered

        def create_info_summary(data):
            info_summary = []
            users = info["user"].unique()
            for user in users:
                dic_user = {}
                data_user = data[data["user"] == user]
                dic_user["user"] = user
                dic_user["user_set"] = data_user.iloc[0]["user_set"]
                dic_user["discarted"] = data_user["discarted"].sum()
                dic_user["calibrate"] = data_user["calibrate"].sum()
                dic_user["fix_removed"] = data_user["fix_removed"].sum()
                dic_user["mean_distance"] = data_user["mean_distance"].mean()
                dic_user["mean_distance_cal"] = data_user["mean_distance_cal"].mean()
                dic_user["mean_distance_10"] = data_user["mean_distance_10"].mean()
                dic_user["mean_distance_cal_10"] = data_user[
                    "mean_distance_cal_10"
                ].mean()
                dic_user["mean_distance_nr"] = data_user["mean_distance"].mean()
                info_summary.append(dic_user)
            info_summary = pd.DataFrame(info_summary)
            return info_summary

        info = create_info_file(folder_path)
        info = info.to_csv(
            folder_path + "/info_summary/info_trials_distance.csv", index=False, sep=";"
        )
        info = pd.read_csv(
            folder_path + "/info_summary/info_trials_distance.csv", sep=";"
        )
        info_ordered = create_info_ordered(info)
        info_ordered.to_csv(
            folder_path + "/info_summary/info_users_trials.csv", sep=";"
        )
        info_summary = create_info_summary(info)
        info_summary.to_csv(
            folder_path + "/info_summary/info_users_distances.csv", sep=";"
        )

    @staticmethod
    def arrange_dimensions(wor_cor, columns_to_calculate):
        shapes = [df.shape[0] for df in wor_cor.values()]
        # Get the index of the least common element
        index_min_shape = [i for i, x in enumerate(shapes) if x == min(shapes)]
        index_max_shape = [i for i, x in enumerate(shapes) if x == max(shapes)]
        # Get the DataFrame with the least common shape
        correct_df = list(wor_cor.values())[index_min_shape[0]]
        list_words_first = correct_df["text"].values.tolist()
        for index in index_max_shape:
            df = list(wor_cor.values())[index]
            list_words_second = df["text"].values.tolist()
            mapped_words_idxs, mapped_words_str = TokenizerAligner().map_words(
                list_words_first, list_words_second
            )
            new_df = {"text": list_words_first}
            for column in list(columns_to_calculate.keys()):
                new_df[column] = TokenizerAligner().map_features_between_paired_list(
                    df[column].values.tolist(),
                    mapped_words_idxs,
                    list_words_first,
                    mode="mean",
                )
            new_df = pd.DataFrame(new_df)
            new_df.index = correct_df.index
            # create normalized columns
            for col, col_n in columns_to_calculate.items():
                new_df[col_n] = new_df[col] / new_df[col].sum()

            wor_cor[list(wor_cor.keys())[index]] = new_df

        return wor_cor

    @staticmethod
    def average_gaze_features_real_participants(path, path_save):
        
        # compute_texts()
        datauserset = EyeTrackingDataUserSet()
        file_path = path + "/info_summary/info_users_trials.csv"
        info = pd.read_csv(
            file_path, sep=";", index_col=0
        )
        users_set = info["user_set"].unique()

        for user_set in users_set:
            user_set_trial_dic = {"all": [], "complete": []}
            # -----------------------------------------------------------------
            path_save_set = path_save + "set_" + str(user_set)
            if not os.path.exists(path_save_set):
                os.makedirs(path_save_set)
            # -----------------------------------------------------------------
            data = info[info["user_set"] == user_set]
            # remove column with all nana
            data = data.dropna(axis=1, how="all")
            trials = data["trial"].unique()
            columns_to_calculate = {
                "fix_duration": "fix_duration_n",
                "first_fix_duration": "first_fix_duration_n",
                "fix_number": "fix_number_n",
            }
            for trial in trials:
                particpants_number = data.columns[2:]
                particpants = data[data["trial"] == trial][
                    particpants_number
                ].columns.tolist()
                wor_cor = {}
                for participant in particpants:
                    data_participant_path = path + "/" + "participant_" + str(participant) + "_" + str(user_set) + "/session_1"
                    data_participant = datauserset.load_words_fixations_trial(
                        data_participant_path, trial
                    )
                    if not isinstance(data_participant, pd.DataFrame):
                        continue
                    missing_col = False
                    for column in list(columns_to_calculate.keys()):
                        if column not in data_participant.columns:
                            missing_col = True
                            break
                    if missing_col is True:
                        continue
                    for col, col_n in columns_to_calculate.items():
                        data_participant[col_n] = (
                            data_participant[col] / data_participant[col].sum()
                        )
                    wor_cor[participant] = data_participant
                    wor_cor[participant] = wor_cor[participant].set_index("number")

                if wor_cor == {}:
                    continue
                # Stack all dataframes on top of each other and then group by the index
                same_dimension = check_same_shape(wor_cor)
                if not same_dimension:
                    print("Different dimensions for user set", user_set, trial)
                    print([x for x in wor_cor.keys()])
                    print([df.shape[0] for df in wor_cor.values()])
                    print(user_set, trial)
                    wor_cor = EyeTrackingAnalyser().arrange_dimensions(
                        wor_cor, columns_to_calculate
                    )
                    print("After correcting", user_set, trial)
                    print([x for x in wor_cor.keys()])
                    print([df.shape[0] for df in wor_cor.values()])
                    print(user_set, trial)
                # change index to text in all dataframes
                columns_to_calculate_list = list(columns_to_calculate.keys())
                columns_to_calculate_list.extend(list(columns_to_calculate.values()))
                combined_df = pd.concat(
                    [df[columns_to_calculate_list] for df in wor_cor.values()]
                )
                # Now calculate the mean for each column, grouping by index to match rows across dataframes
                wor_cor_all = combined_df.groupby(combined_df.index).mean()
                wor_cor_all["text"] = list(wor_cor.values())[0]["text"]
                datauserset.save_words_fixations_trial(
                    path_save_set, trial, wor_cor_all
                )
                user_set_trial_dic["all"].append(trial)
                if len(wor_cor) == 3:
                    user_set_trial_dic["complete"].append(trial)
            save_to_json(
                user_set_trial_dic, filename=path_save_set + "/info_trials.json"
            )

    # @staticmethod
    # def compute_entropy(users, sessions, path=None):
    #     fixations_entropy = []
    #     for user in users:
    #         for session in sessions:
    #             try:
    #                 # -------------------- asign fixations to words and save them --------------------
    #                 analyzer = EyeTrackingDataImage(
    #                     user=user, session=session, path=path
    #                 )
    #                 fixations_entropy.extend(analyzer.compute_entropy_all())
    #             except Exception as e:
    #                 print(e)
    #     fixations_entropy = pd.DataFrame(fixations_entropy)
    #     return fixations_entropy

    @staticmethod
    def compute_metrics(users, sessions, path=None):
        metrics = pd.DataFrame()
        for user in users:
            for session in sessions:
                try:
                    # -------------------- asign fixations to words and save them --------------------
                    analyzer = EyeTrackingDataImage(
                        user=user, session=session, path=path
                    )
                    metrics = pd.concat([metrics, analyzer.data])
                except Exception as e:
                    print(e)
        return metrics

    @staticmethod
    def compute_fixations_and_save(users, sessions, path=None):
        """
        Compute fixations and save them on a CSV
        """
        if isinstance(users, list):
            users = {user: user for user in users}
        for user, user_set in users.items():
            for session in sessions:
                # try:
                # -------------------- asign fixations to words and save them --------------------
                analyzer = EyeTrackingDataImage(
                    user=user, user_set=user_set, session=session, path=path
                )
                fixations_all, words_fix_all = analyzer.asign_fixations_words_all()
                analyzer.save_fixations(words_fix_all, fixations_all)
                # except Exception as e:
                #     print(e)

    @staticmethod
    def compute_fixations_process_and_save(users, sessions, path=None):
        """
        Compute fixations and save them on a CSV
        """
        if isinstance(users, list):
            users = {user: user for user in users}
        for user, user_set in users.items():
            for session in sessions:
                # -------------------- asign fixations to words and save them --------------------
                analyzer = EyeTrackingDataImage(
                    user=user, user_set=user_set, session=session, path=path
                )
                (
                    fixations_all,
                    words_fix_all,
                    info,
                ) = analyzer.asign_fixations_process_words_all()

                analyzer.save_fixations(words_fix_all, fixations_all, info)

    @staticmethod
    def filter_entropy_v0(fixation_entropy, entropy_users, user):
        entropy_chosen = [
            entropy_value
            for trial, entropy_value in fixation_entropy.items()
            if str(trial).endswith(".1")
        ]
        entropy_rejected = [
            entropy_value
            for trial, entropy_value in fixation_entropy.items()
            if not str(trial).endswith(".1")
        ]
        entropy_users.append(
            {
                "Condition": "Chosen",
                "Entropy": np.mean(entropy_chosen),
                "Entropy_std": np.std(entropy_chosen, ddof=1),
                "user": user,
            }
        )

        entropy_users.append(
            {
                "Condition": "Rejected",
                "Entropy": np.mean(entropy_rejected),
                "Entropy_std": np.std(entropy_rejected, ddof=1),
                "user": user,
            }
        )
        return entropy_users

    @staticmethod
    def filter_entropy(fixation_entropy):
        entropy_chosen = [
            {
                "Entropy": entropy_value,
                "trial": trial,
                "Condition": "chosen",
                "user": user,
            }
            for trial, entropy_value, user in fixation_entropy
            if str(trial).endswith(".1")
        ]
        entropy_rejected = [
            {
                "Entropy": entropy_value,
                "trial": trial,
                "Condition": "rejected",
                "user": user,
            }
            for trial, entropy_value, user in fixation_entropy
            if not str(trial).endswith(".1")
        ]
        # entropy_all = entropy_chosen.extend(entropy_rejected)
        return entropy_chosen + entropy_rejected

    @staticmethod
    def normalize_entropy(data):
        entropy = [value for _, value, _ in data]
        # Find the maximum value of the second values
        max_value = max(entropy)
        # Normalize second values and create a new tuple
        normalized_data = [
            (first, value / max_value, user) for first, value, user in data
        ]
        return normalized_data

    def compute_entropy_and_save(self, users, path=None, entropy_name="fuzzy"):
        """
        Compute fixations and save them on a CSV
        """
        if isinstance(users, list):
            users = {user: user for user in users}
        # entropy_users_x = []
        # entropy_users_y = []
        session = 1
        fixation_entropy_x = []
        fixation_entropy_y = []
        for user, user_set in users.items():
            # -------------------- asign fixations to words and save them --------------------
            analyzer = EyeTrackingDataImage(
                user=user, user_set=user_set, session=session, path=path
            )
            trials = [
                float(x)
                for x in analyzer.fixations_raw["USER"].unique()
                if len(str(x).split(".")) > 1
            ]
            for trial in trials:
                # num_bins = 100  # Number of bins along each axis
                fixations_trial_original = analyzer._get_fixations_trial(trial)
                try:
                    n_words = len(
                        analyzer.data[analyzer.data["n_resp"] == str(trial)]
                        .iloc[0]["resp_text"]
                        .split(" ")
                    )
                except Exception as e:
                    try:
                        n_words = len(
                            analyzer.data[analyzer.data["n_resp"] == float(trial)]
                            .iloc[0]["resp_text"]
                            .split(" ")
                        )
                    except Exception as e:
                        print(f"mal {trial}")
                        continue
                print(f"bien {trial}")
                # width = analyzer.x_screen
                # height = analyzer.y_screen

                # try permutation 2 D
                # fixation_entropy[trial] = self.permutation_entropy_2d(
                #     fixations_trial_original[['x', 'y']].to_numpy(), m=m, delay=delay
                # )
                if entropy_name == "fuzzy":
                    r = 0.1  # Define your tolerance level
                    m = 2  # Define embedding dimension
                    fixation_entropy_x.append(
                        (
                            trial,
                            self.compute_fuzzy_entropy(
                                fixations_trial_original[["x"]].to_numpy().flatten(),
                                m,
                                r,
                            )
                            / n_words,
                            user,
                        )
                    )
                    fixation_entropy_y.append(
                        (
                            trial,
                            self.compute_fuzzy_entropy(
                                fixations_trial_original[["y"]].to_numpy().flatten(),
                                m,
                                r,
                            )
                            / n_words,
                            user,
                        )
                    )
                else:
                    m = 3  # Embedding dimension
                    delay = 1
                    fixation_entropy_x.append(
                        (
                            trial,
                            self.permutation_entropy(
                                fixations_trial_original[["x"]].to_numpy().flatten(),
                                order=3,
                            )
                            / n_words,
                            user,
                        )
                    )
                    fixation_entropy_y.append(
                        (
                            trial,
                            self.permutation_entropy(
                                fixations_trial_original[["y"]].to_numpy().flatten(),
                                order=3,
                            )
                            / n_words,
                            user,
                        )
                    )

        # fixation_entropy_x = self.normalize_entropy(fixation_entropy_x)
        # fixation_entropy_y = self.normalize_entropy(fixation_entropy_y)
        fixation_entropy_x = self.filter_entropy(fixation_entropy_x)
        fixation_entropy_y = self.filter_entropy(fixation_entropy_y)

        fixation_entropy_x = pd.DataFrame(fixation_entropy_x)
        fixation_entropy_y = pd.DataFrame(fixation_entropy_y)

        self.plot_entropy(fixation_entropy_x, "x", entropy_name, path=path)
        self.plot_entropy(fixation_entropy_y, "y", entropy_name, path=path)

        fixation_entropy_x.to_csv(
            path + "/entropy/" + "entropy_" + entropy_name + "_" + "x" + ".csv",
            sep=";",
            index=False,
        )
        fixation_entropy_y.to_csv(
            path + "/entropy/" + "entropy_" + entropy_name + "_" + "y" + ".csv",
            sep=";",
            index=False,
        )
        return fixation_entropy_x, fixation_entropy_y

    def load_entropy_and_save(self, path=None, entropy_name="fuzzy"):
        fixation_entropy_x = pd.read_csv(
            path + "/entropy/" + "entropy_" + entropy_name + "_" + "x" + ".csv", sep=";"
        )
        fixation_entropy_y = pd.read_csv(
            path + "/entropy/" + "entropy_" + entropy_name + "_" + "y" + ".csv", sep=";"
        )
        # fixation_entropy_duration = pd.read_csv(
        #     path + "/entropy/" + "entropy_" + entropy_name + "_" + "duration" + ".csv",
        #     sep=";",
        # )
        p_corrected_x = self.compute_posthoc_comparisons_entropy(fixation_entropy_x)
        p_corrected_y = self.compute_posthoc_comparisons_entropy(fixation_entropy_y)
        # self.compute_posthoc_comparisons(fixation_entropy_duration)
        self.plot_entropy(
            fixation_entropy_x, "x", entropy_name, path=path, p_values=p_corrected_x
        )
        self.plot_entropy(
            fixation_entropy_y, "y", entropy_name, path=path, p_values=p_corrected_y
        )
        # self.plot_entropy(fixation_entropy_duration, 'duration', entropy_name, path=path)
        return fixation_entropy_x, fixation_entropy_y  # , fixation_entropy_duration

    @staticmethod
    def compute_posthoc_comparisons_entropy(df):
        def filter_dataframe_trials(df):
            df["prefix"] = df["trial"].astype(str).str.split(".").str[0]
            prefix_counts = Counter(df["prefix"])
            # Filter rows where the prefix appears at least twice
            df_filtered = df[df["prefix"].map(prefix_counts) >= 2].drop(
                columns="prefix"
            )
            # df_filtered = df_filtered.drop(columns='prefix')

            return df_filtered

        t_stats = []
        p_values = []
        for user in df["user"].unique():
            df_user = df[df["user"] == user]
            df_user = filter_dataframe_trials(df_user)
            chosen = df_user[df["Condition"] == "chosen"]["Entropy"]
            rejected = df_user[df["Condition"] == "rejected"]["Entropy"]
            t_stat, p_val = ttest_rel(chosen, rejected)
            t_stats.append(t_stat)
            p_values.append(p_val)

        # Apply multiple comparisons correction (e.g., Bonferroni)
        _, p_corrected, _, _ = multipletests(p_values, method="fdr_tsbh")

        # Create a DataFrame to display the results
        results = pd.DataFrame(
            {
                "user": df["user"].unique(),
                "t_statistic": t_stats,
                "p_value_uncorrected": p_values,
                "p_value_corrected": p_corrected,
            }
        )

        print(results)
        return p_corrected

    @staticmethod
    def plot_entropy(entropy_users_df, coordinate, entropy_name, path, p_values=[]):
        palette_colors = [
            "#8cc5e3",  # Original Blue
            "#1a80bb",  # Lighter Blue
        ]
        y_label = "Entropy"
        fig, axes = plt.subplots(figsize=(18, 6))
        entropy_users_df["user"] = entropy_users_df["user"].replace(
            {
                3: 2,
                4: 3,
                5: 4,
                6: 5,
                7: 6,
                8: 7,
                9: 8,
                10: 9,
                12: 10,
                13: 11,
                14: 12,
                15: 13,
                16: 14,
                17: 15,
                18: 16,
                19: 17,
                20: 18,
                21: 19,
                22: 20,
                23: 21,
                24: 22,
                25: 23,
                26: 24,
            }
        )
        sns.boxplot(
            hue="Condition",
            y=y_label,
            data=entropy_users_df,
            palette=palette_colors,
            showfliers=False,
            x="user",
            boxprops=dict(edgecolor="black"),
            whiskerprops=dict(color="black"),
            capprops=dict(color="black"),
            medianprops=dict(color="black"),
        )
        for i, p_value in enumerate(p_values):
            if p_value < 0.001:
                # Add an asterisk above the user's boxplot
                plt.text(
                    i - 1,
                    0.48,
                    "***",
                    ha="center",
                    va="bottom",
                    color="black",
                    fontsize=10,
                )
            elif p_value < 0.01:
                # Add an asterisk above the user's boxplot
                plt.text(
                    i - 1,
                    0.48,
                    "**",
                    ha="center",
                    va="bottom",
                    color="black",
                    fontsize=10,
                )
            elif p_value < 0.05:
                # Add an asterisk above the user's boxplot
                plt.text(
                    i - 1,
                    0.48,
                    "*",
                    ha="center",
                    va="bottom",
                    color="black",
                    fontsize=10,
                )
        # Add title and labels (optional)
        plt.title("Boxplot of Entropy values in coordinate " + coordinate)
        plt.xlabel("User")
        plt.ylabel(y_label)
        # Save the figure
        path_save = path + "/entropy/"
        # check if path_save exists
        if not os.path.exists(path_save):
            os.makedirs(path_save)
        plt.savefig(
            path_save + "boxplot_" + entropy_name + "_" + coordinate + ".png",
            dpi=300,
            bbox_inches="tight",
        )  # Save as PNG
        # plt.savefig(path_save + 'boxplot_' + entropy_name+ '_' + coordinate + '.pdf', bbox_inches='tight')           # Save as PDF (optional)

        # Show the plot (optional)
        plt.show()

    @staticmethod
    def compute_images_and_save(users, sessions, path=None, calibrate=False):
        if isinstance(users, list):
            users = {user: user for user in users}
        for user, user_set in users.items():
            for session in sessions:
                try:
                    # -------------------- asign fixations to words and save then on the image--------------------
                    analyzer = EyeTrackingDataImage(
                        user=user, user_set=user_set, session=session, path=path
                    )
                    # we obtain all trials of this user and session
                    # trials = [827.1]
                    for trial in analyzer.trials:
                        print(trial)
                        (
                            fixations_trial,
                            words_fix_trial,
                            total_distance,
                        ) = analyzer.asign_fixations_words_trial(trial)
                        analyzer.plot_image_trial_colors(
                            trial,
                            fixations_trial=fixations_trial,
                            words_fix_trial=words_fix_trial,
                            fixations=True,
                            coordinates=True,
                            save=True,
                            calibrate=calibrate,
                        )
                except Exception as e:
                    print(e)

    @staticmethod
    def compute_images_and_show(
        users,
        sessions,
        path=None,
        calibrate=False,
        plot_fixations_words=True,
        print_fix_distance=False,
        filter_proportion=0,
        filter_value=0,
    ):
        if isinstance(users, list):
            users = {user: user for user in users}
        for user, user_set in users.items():
            for session in sessions:
                try:
                    # -------------------- asign fixations to words and print them on the image --------------------
                    analyzer = EyeTrackingDataImage(
                        user=user, user_set=user_set, session=session, path=path
                    )
                    # we obtain all trials of this user and session
                    # trials = [827.1]
                    for trial in analyzer.trials:
                        if calibrate:
                            (
                                words_fix_trial,
                                fixations_trial,
                                total_distance,
                                fixations_trial_remove,
                                words_fix_trial_cal,
                                fixations_trial_cal,
                                total_distance_cal,
                                fixations_trial_remove_cal,
                            ) = analyzer.asign_fixations_words_trial(
                                trial,
                                check_calibration=True,
                                filter_proportion=filter_proportion,
                                filter_value=filter_value,
                                return_notcalibrated=True,
                                return_removed=True,
                            )

                        else:
                            (
                                words_fix_trial_cal,
                                fixations_trial_cal,
                                total_distance_cal,
                                fixations_trial_remove_cal,
                            ) = None, None, None, []
                            (
                                words_fix_trial,
                                fixations_trial,
                                total_distance,
                                fixations_trial_remove,
                            ) = analyzer.asign_fixations_words_trial(
                                trial,
                                check_calibration=False,
                                filter_proportion=filter_proportion,
                                filter_value=filter_value,
                                return_notcalibrated=False,
                                return_removed=True,
                            )

                        if plot_fixations_words:
                            for idx, row in words_fix_trial.iterrows():
                                analyzer.plot_image_trial_colors(
                                    trial,
                                    fixations_trial=fixations_trial,
                                    fixations_trial_cal=fixations_trial_cal,
                                    words_fix_trial=words_fix_trial,
                                    fix_diff=row["fixations"],
                                    words_diff=idx,
                                    fixations=True,
                                    coordinates=True,
                                    calibrate=calibrate,
                                    ids_removed=fixations_trial_remove,
                                    ids_removed_cal=fixations_trial_remove_cal,
                                    print_fix_distance=print_fix_distance,
                                )
                        else:
                            analyzer.plot_image_trial_colors(
                                trial,
                                fixations_trial=fixations_trial,
                                fixations_trial_cal=fixations_trial_cal,
                                words_fix_trial=words_fix_trial,
                                fix_diff=fixations_trial_remove,
                                fixations=True,
                                coordinates=True,
                                calibrate=calibrate,
                                ids_removed=fixations_trial_remove,
                                ids_removed_cal=fixations_trial_remove_cal,
                                print_fix_distance=print_fix_distance,
                                save=True,
                            )

                            # analyzer.plot_image_trial_colors(trial, fixations_trial=fixations_trial, words_fix_trial=words_fix_trial, fixations=True, coordinates=True, calibrate=False)
                        # analyzer.save_fixations({trial:words_fix_trial})
                except Exception as e:
                    print(e)

    @staticmethod
    def compute_images_and_show_calibrate(
        users,
        sessions,
        path=None,
        print_fix_distance=False,
        filter_proportion=0,
    ):
        if isinstance(users, list):
            users = {user: user for user in users}
        for user, user_set in users.items():
            for session in sessions:
                try:
                    # -------------------- asign fixations to words and print them on the image --------------------
                    analyzer = EyeTrackingDataImage(
                        user=user, user_set=user_set, session=session, path=path
                    )
                    # we obtain all trials of this user and session
                    # trials = [827.1]
                    for trial in analyzer.trials:
                        (
                            words_fix_trial,
                            fixations_trial,
                            total_distance,
                            fixations_trial_remove,
                            words_fix_trial_cal,
                            fixations_trial_cal,
                            total_distance_cal,
                            fixations_trial_remove_cal,
                        ) = analyzer.asign_fixations_words_trial(
                            trial,
                            check_calibration=True,
                            filter_proportion=filter_proportion,
                            return_all=True,
                        )

                        analyzer.plot_image_trial_colors(
                            trial,
                            fixations_trial=fixations_trial,
                            fixations_trial_cal=fixations_trial_cal,
                            words_fix_trial=words_fix_trial,
                            fix_diff=fixations_trial_remove,
                            fixations=True,
                            coordinates=True,
                            calibrate=True,
                            total_distance=total_distance,
                            total_distance_cal=total_distance_cal,
                            print_fix_distance=print_fix_distance,
                        )

                except Exception as e:
                    print(e)

    @staticmethod
    def compute_images_and_show_removed(
        users,
        sessions,
        path=None,
        print_fix_distance=False,
        plot_fixations_words=False,
        calibrate=True,
        filter_proportion=0,
        filter_value=0,
    ):
        if isinstance(users, list):
            users = {user: user for user in users}
        for user, user_set in users.items():
            for session in sessions:
                # try:
                # -------------------- asign fixations to words and print them on the image --------------------
                analyzer = EyeTrackingDataImage(
                    user=user, user_set=user_set, session=session, path=path
                )
                # we obtain all trials of this user and session
                # trials = [827.1]
                for trial in analyzer.trials:
                    if calibrate:
                        (
                            words_fix_trial,
                            fixations_trial,
                            total_distance,
                            fixations_trial_remove,
                        ) = analyzer.asign_fixations_words_trial(
                            trial,
                            check_calibration=calibrate,
                            filter_proportion=filter_proportion,
                            return_notcalibrated=False,
                            return_removed=True,
                        )
                    analyzer.plot_image_trial_colors(
                        trial,
                        fixations_trial=fixations_trial,
                        fixations_trial_cal=None,
                        words_fix_trial=words_fix_trial,
                        fix_diff="",
                        fixations=True,
                        coordinates=True,
                        calibrate=False,
                        ids_removed=fixations_trial_remove,
                        print_fix_distance=print_fix_distance,
                    )

                # except Exception as e:
                #     print(e)

    @staticmethod
    def compute_general_features(users, sessions, path=None):
        for user in users:
            for session in sessions:
                try:
                    # -------------------- asign fixations to words and print them on the image --------------------
                    analyzer = EyeTrackingDataImage(
                        user=user, session=session, path=path
                    )
                    # we obtain all trials of this user and session
                    # trials = [827.1]
                    features = []
                    for trial in analyzer.trials:
                        print(trial)
                        (
                            fixations_trial,
                            words_fix_trial,
                            total_distance,
                        ) = analyzer.asign_fixations_words_trial(trial)
                        features_trial = analyzer.compute_general_features_trial(
                            fixations_trial, words_fix_trial
                        )
                        features_trial["trial"] = trial
                        features_trial["user"] = user
                        features_trial["session"] = session
                        features.append(features_trial)
                    analyzer.save_features(features)

                # except NoUserSessionData as e:
                #     print(e)
                except Exception as e:
                    print(e)

    @staticmethod
    def compute_n_instances(path="oasstetc/gaze_features_real/"):
        n_instances_user_set = {}
        for user_set in list(range(1, 9)):
            with open(path + "set_" + str(user_set) + "/info_trials.json", "r") as file:
                data_info = json.load(file)
            n_all = len(data_info["all"])
            n_complete = len(data_info["complete"])
            chosen_complete = [
                x for x in data_info["complete"] if str(x).endswith(".1")
            ]
            rejected_complete = [
                x for x in data_info["complete"] if not str(x).endswith(".1")
            ]
            n_chosen_complete = len(chosen_complete)
            n_rejected_complete = len(rejected_complete)
            chosen_all = [x for x in data_info["all"] if str(x).endswith(".1")]
            rejected_all = [x for x in data_info["all"] if not str(x).endswith(".1")]
            n_chosen_all = len(chosen_all)
            n_rejected_all = len(rejected_all)
            n_instances_user_set[user_set] = {
                "all": n_all,
                "chosen_all": n_chosen_all,
                "rejected_all": n_rejected_all,
                "complete": n_complete,
                "chosen_complete": n_chosen_complete,
                "rejected_complete": n_rejected_complete,
            }
        n_instances_user_set = pd.DataFrame(n_instances_user_set)
        n_instances_user_set = n_instances_user_set.T
        n_instances_user_set.to_csv(
            "/".join(path.split("/")[:-2])
            + "/raw_data/info_summary/n_instances_user_set.csv",
            sep=";",
        )
