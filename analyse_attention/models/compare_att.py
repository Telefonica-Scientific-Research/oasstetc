import scipy
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from models.human_att import HumanAttentionExtractor
from models.model_att import ModelAttentionExtractor
from scipy.stats import spearmanr as spearmanr
import pathlib
import re
import os
from matplotlib.colors import LinearSegmentedColormap
import matplotlib.pyplot as plt
from statsmodels.stats.multitest import multipletests
from collections import Counter
from scipy.stats import ttest_rel


class CompareAttention:
    def __init__(self, model_name, model_type, path):
        self.model_name = model_name
        self.path = path
        self.model_type = model_type
        self.gaze_features_names = {
            "fix_duration_n": "TRT",
            "fix_duration": "TRT",
            "first_fix_duration": "FFD",
            "first_fix_duration_n": "FFD",
            "fix_number": "nFix",
        }

    @staticmethod
    def compare_between_models_per_userset(
        folder,
        gaze_feature="fix_duration_n",
        filter_completed=False,
    ):
        folder_filter = "completed" if filter_completed else "not_filtered"
        if isinstance(folder, str):
            folder = pathlib.Path(folder)
        results_folder = "results"
        file_paths = list(folder.rglob("*"))
        text = "correlation_userset_" + str(gaze_feature) + ".csv"
        pattern = re.escape(text)
        data_models = {}
        for file in file_paths:
            match = re.search(pattern, str(file))
            if match and folder_filter in str(file) and results_folder not in str(file):
                print(file)
                data = pd.read_csv(file, sep=";", index_col=0)
                data_models[str(file).split("/")[-3]] = data.max()

        # create row with the mean of all rows
        data_models = pd.DataFrame(data_models)
        mean = data_models.mean()
        # create row with the std of all rows
        std = data_models.std()
        data_models.loc["mean"] = mean
        data_models.loc["std"] = std
        folder_to_save = str(folder) + "/" + results_folder + "/" + str(folder_filter)
        if not os.path.exists(folder_to_save):
            os.makedirs(folder_to_save)
        data_models.to_csv(
            folder_to_save + "/correlation_userset_" + str(gaze_feature) + ".csv",
            sep=";",
        )

    @staticmethod
    def compare_between_models_per_trials(
        folder,
        gaze_feature="fix_duration_n",
        filter_completed=False,
    ):
        folder_filter = "completed" if filter_completed else "not_filtered"
        if isinstance(folder, str):
            folder = pathlib.Path(folder)
        results_folder = "results"
        file_paths = list(folder.rglob("*"))
        text = folder_filter + "/correlation_trials_" + str(gaze_feature) + ".csv"
        pattern = re.escape(text)
        data_models = {}
        counter = 1
        for file in file_paths:
            match = re.search(pattern, str(file))
            if (
                match
                and results_folder not in str(file)
                and "chosen" not in str(file)
                and "rejected" not in str(file)
            ):
                print(counter, ":", file)
                data = pd.read_csv(file, sep=";", index_col=0)
                data_models[str(file).split("/")[-3]] = data.loc[data["mean"].idxmax()]
                counter += 1

        # create row with the mean of all rows
        data_models = pd.DataFrame(data_models)
        folder_to_save = str(folder) + "/" + results_folder + "/" + str(folder_filter)
        if not os.path.exists(folder_to_save):
            os.makedirs(folder_to_save)
        data_models.to_csv(
            folder_to_save + "/correlation_trials_" + str(gaze_feature) + ".csv",
            sep=";",
        )

    @staticmethod
    def compare_between_models_chosenrejected_per_userset(
        folder, gaze_feature="fix_duration_n", filter_completed=False
    ):
        folder_filter = "completed" if filter_completed else "not_filtered"
        results_folder = "results"
        tags_chosen_rejected = ["chosen", "rejected"]
        for tag_chosen_rejected in tags_chosen_rejected:
            if isinstance(folder, str):
                folder = pathlib.Path(folder)
            file_paths = list(folder.rglob("*"))
            text = (
                tag_chosen_rejected
                + "/"
                + folder_filter
                + "/"
                + "correlation_userset_"
                + str(gaze_feature)
                + ".csv"
            )
            pattern = re.escape(text)
            data_models = {}
            for file in file_paths:
                match = re.search(pattern, str(file))
                if match and results_folder not in str(file):
                    print(file)
                    data = pd.read_csv(file, sep=";", index_col=0)
                    data_models[str(file).split("/")[-4]] = data.max()
            # create row with the mean of all rows
            data_models = pd.DataFrame(data_models)
            mean = data_models.mean()
            # create row with the std of all rows
            std = data_models.std()
            data_models.loc["mean"] = mean
            data_models.loc["std"] = std
            folder_tag = (
                str(folder)
                + "/"
                + results_folder
                + "/"
                + str(tag_chosen_rejected)
                + "/"
                + str(folder_filter)
                + "/"
            )
            if not os.path.exists(folder_tag):
                os.makedirs(folder_tag)
            data_models.to_csv(
                folder_tag + "correlation_userset_" + str(gaze_feature) + ".csv",
                sep=";",
            )

    @staticmethod
    def compare_between_models_chosenrejected_per_trials(
        folder, gaze_feature="fix_duration_n", filter_completed=False
    ):
        folder_filter = "completed" if filter_completed else "not_filtered"
        results_folder = "results"
        tags_chosen_rejected = ["chosen", "rejected"]
        for tag_chosen_rejected in tags_chosen_rejected:
            if isinstance(folder, str):
                folder = pathlib.Path(folder)
            file_paths = list(folder.rglob("*"))
            text = (
                tag_chosen_rejected
                + "/"
                + folder_filter
                + "/"
                + "correlation_trials_"
                + str(gaze_feature)
                + ".csv"
            )
            pattern = re.escape(text)
            data_models, data_models_all = {}, {}
            counter = 1
            for file in file_paths:
                match = re.search(pattern, str(file))
                if match and results_folder not in str(file):
                    print(counter, ":", file)
                    data = pd.read_csv(file, sep=";", index_col=0)
                    data_all = pd.read_csv(
                        str(file).replace(
                            "correlation_trials_", "correlation_trials_all"
                        ),
                        sep=";",
                        index_col=0,
                    )
                    max_layer = data["mean"].idxmax()
                    data_models[str(file).split("/")[-4]] = data.loc[max_layer]
                    data_models_all[str(file).split("/")[-4]] = data_all.loc[max_layer]
                    counter += 1
            # create row with the mean of all rows
            data_models = pd.DataFrame(data_models)
            data_models_all = pd.DataFrame(data_models_all)

            folder_tag = (
                str(folder)
                + "/"
                + results_folder
                + "/"
                + str(tag_chosen_rejected)
                + "/"
                + str(folder_filter)
                + "/"
            )
            if not os.path.exists(folder_tag):
                os.makedirs(folder_tag)
            data_models.to_csv(
                folder_tag + "correlation_trials_" + str(gaze_feature) + ".csv",
                sep=";",
            )
            data_models_all.to_csv(
                folder_tag + "correlation_trials_alldata" + str(gaze_feature) + ".csv",
                sep=";",
            )

    @staticmethod
    def compute_posthoc_comparisons_correlation(df_chosen, df_rejected):
        t_stats = []
        p_values = []
        for model in df_chosen.columns.to_list():
            chosen = df_chosen[model]
            rejected = df_rejected[model]
            # if len(chosen) > len(rejected):
            #     chosen = chosen[: len(rejected)]
            # elif len(chosen) < len(rejected):
            #     rejected = rejected[: len(chosen)]
            # if len(chosen) != len(rejected):
            #     print("Error: The number of trials is not the same")

            mask = ~np.isnan(chosen) & ~np.isnan(rejected)
            x_clean = chosen[mask]
            y_clean = rejected[mask]
            t_stat, p_val = ttest_rel(x_clean, y_clean)
            t_stats.append(t_stat)
            p_values.append(p_val)

        # Apply multiple comparisons correction (e.g., Bonferroni)
        _, p_corrected, _, _ = multipletests(p_values, method="fdr_tsbh")

        # Create a DataFrame to display the results
        results = pd.DataFrame(
            {
                "model": df_chosen.columns.to_list(),
                "t_statistic": t_stats,
                "p_value_uncorrected": p_values,
                "p_value_corrected": p_corrected,
            }
        )

        print(results)
        return dict(zip(df_chosen.columns.to_list(), p_corrected))

    @staticmethod
    def estatistical_analyse_chosenrejected_per_trials(
        folder, gaze_feature="fix_duration_n", filter_completed=False
    ):
        folder_filter = "completed" if filter_completed else "not_filtered"
        results_folder = "results"
        tags_chosen_rejected = ["chosen", "rejected"]
        for tag_chosen_rejected in tags_chosen_rejected:
            if isinstance(folder, str):
                folder = pathlib.Path(folder)
            file_paths = list(folder.rglob("*"))
            text = (
                tag_chosen_rejected
                + "/"
                + folder_filter
                + "/"
                + "correlation_trials_"
                + str(gaze_feature)
                + ".csv"
            )
            pattern = re.escape(text)
            data_models = {}
            counter = 1
            for file in file_paths:
                match = re.search(pattern, str(file))
                if match and results_folder not in str(file):
                    print(counter, ":", file)
                    data = pd.read_csv(file, sep=";", index_col=0)
                    max_layer = data["mean"].idxmax()
                    data_models[str(file).split("/")[-4]] = data.loc[max_layer]
                    counter += 1
            # create row with the mean of all rows
            data_models = pd.DataFrame(data_models)

            folder_tag = (
                str(folder)
                + "/"
                + results_folder
                + "/"
                + str(tag_chosen_rejected)
                + "/"
                + str(folder_filter)
                + "/"
            )
            if not os.path.exists(folder_tag):
                os.makedirs(folder_tag)
            data_models.to_csv(
                folder_tag + "correlation_trials_" + str(gaze_feature) + ".csv",
                sep=";",
            )

    def compute_sc_layer(
        self,
        gaze_features_layer: dict,
        model_attention_layer: dict,
        gaze_feature: str = "fix_duration_n",
        layer: int = 0,
        filter_trials: list = [],
    ):
        spearman_trial = []
        for trial in list(gaze_features_layer.keys()):
            if float(trial) in model_attention_layer.keys():
                # Compute Spearman correlation
                if len(filter_trials) > 0 and float(trial) not in filter_trials:
                    continue
                sc, p_value = spearmanr(
                    gaze_features_layer[trial][gaze_feature].values,
                    model_attention_layer[float(trial)][layer]["attention"].values,
                )
                spearman_trial.append(sc)
        # order the spearman_trial list by valeus and remove the first 5% of values
        spearman_trial = sorted(spearman_trial)
        spearman_trial = spearman_trial[int(len(spearman_trial) * 0.05) :]

        mean_spearman = np.nanmean(np.asarray(spearman_trial))
        std_spearman = np.nanstd(np.asarray(spearman_trial))

        return mean_spearman, std_spearman, spearman_trial

    def compute_sc_model_per_userset(
        self,
        gaze_feature="fix_duration_n",
        save=True,
        filter_completed=False,
        folder_attention="attention",
    ):
        folder_filter = "completed" if filter_completed else "not_filtered"
        sc_users = {}
        for user_set in range(1, 9):
            sc_users[user_set] = self.compute_sc_user_set(
                user_set=user_set,
                gaze_feature=gaze_feature,
                filter_completed=filter_completed,
                folder_attention=folder_attention,
            )
        df = pd.DataFrame(sc_users)
        # order by index DESC
        df.sort_index(ascending=False, inplace=True)
        # Transpose the DataFrame to match the row-column structure
        if save:
            folder_to_save = (
                str(self.path)
                + folder_attention
                + "/"
                + str(self.model_name.split("/")[1])
                + str("/")
                + str(folder_filter)
            )
            if not os.path.exists(folder_to_save):
                os.makedirs(folder_to_save)
            df.to_csv(
                folder_to_save + "/correlation_userset_" + str(gaze_feature) + ".csv",
                sep=";",
            )

        return sc_users

    def compute_sc_model_per_trial(
        self,
        gaze_feature="fix_duration_n",
        save=True,
        filter_completed=False,
        folder_attention="attention",
    ):
        folder_filter = "completed" if filter_completed else "not_filtered"

        sc_users_all, _ = self.compute_sc_all_userset(
            gaze_feature=gaze_feature,
            filter_completed=filter_completed,
        )
        df = pd.DataFrame(sc_users_all).T
        df.rename(columns={df.columns[0]: "mean", df.columns[1]: "std"}, inplace=True)
        # order by index DESC
        df.sort_index(ascending=False, inplace=True)
        # Transpose the DataFrame to match the row-column structure
        if save:
            folder_to_save = (
                str(self.path)
                + folder_attention
                + "/"
                + str(self.model_name.split("/")[1])
                + str("/")
                + str(folder_filter)
            )
            if not os.path.exists(folder_to_save):
                os.makedirs(folder_to_save)
            df.to_csv(
                folder_to_save + "/correlation_trials_" + str(gaze_feature) + ".csv",
                sep=";",
            )

        return sc_users_all

    def compute_sc_model_chosenrejected_per_trials(
        self,
        gaze_feature="fix_duration_n",
        save=True,
        filter_completed=False,
        folder_attention="attention",
    ):
        folder_filter = "completed" if filter_completed else "not_filtered"
        tags_chosen_rejected = ["chosen", "rejected"]
        for tag_chosen_rejected in tags_chosen_rejected:
            sc_users_all, sc_users_all_data = self.compute_sc_all_userset(
                gaze_feature=gaze_feature,
                filter_completed=filter_completed,
                filter_cr=tag_chosen_rejected,
                folder_attention=folder_attention,
            )
            df = pd.DataFrame(sc_users_all).T
            df_all = pd.DataFrame(sc_users_all_data).T
            df.rename(
                columns={df.columns[0]: "mean", df.columns[1]: "std"}, inplace=True
            )
            # order by index DESC
            df.sort_index(ascending=False, inplace=True)
            # Transpose the DataFrame to match the row-column structure
            if save:
                folder_to_save = (
                    str(self.path)
                    + folder_attention
                    + str("/")
                    + str(self.model_name.split("/")[1])
                    + "/"
                    + tag_chosen_rejected
                    + str("/")
                    + str(folder_filter)
                )

                if not os.path.exists(folder_to_save):
                    os.makedirs(folder_to_save)
                df.to_csv(
                    folder_to_save
                    + "/correlation_trials_"
                    + str(gaze_feature)
                    + ".csv",
                    sep=";",
                )
                df_all.to_csv(
                    folder_to_save
                    + "/correlation_trials_all"
                    + str(gaze_feature)
                    + ".csv",
                    sep=";",
                )

        return sc_users_all

    def compute_sc_model_chosenrejected_per_userset(
        self,
        gaze_feature="fix_duration_n",
        save=True,
        tag_text="",
        filter_completed=False,
        folder_attention="attention",
    ):
        tags_chosen_rejected = ["chosen", "rejected"]
        folder_filter = "completed" if filter_completed else "not_filtered"
        for tag_chosen_rejected in tags_chosen_rejected:
            sc_users = {}
            for user_set in range(1, 9):
                sc_users[user_set] = self.compute_sc_user_set(
                    user_set=user_set,
                    gaze_feature=gaze_feature,
                    filter_completed=filter_completed,
                    filter_cr=tag_chosen_rejected,
                    folder_attention=folder_attention,
                )
            df = pd.DataFrame(sc_users)
            # order by index DESC
            df.sort_index(ascending=False, inplace=True)
            # Transpose the DataFrame to match the row-column structure
            folder_to_save = (
                self.path
                + folder_attention
                + "/"
                + self.model_name.split("/")[1]
                + "/"
                + tag_chosen_rejected
                + "/"
                + folder_filter
                + "/"
            )
            # check if path exists
            if not os.path.exists(folder_to_save):
                os.makedirs(folder_to_save)
            if save:
                df.to_csv(
                    folder_to_save
                    + "correlation_userset_"
                    + str(gaze_feature)
                    + str(tag_text)
                    + ".csv",
                    sep=";",
                )

        return sc_users

    def compute_sc_all_userset(
        self,
        gaze_feature="fix_duration_n",
        filter_completed=False,
        filter_cr=False,
        folder_attention="attention",
    ):
        # Load the model
        attention_trials, real_gaze_fixations, filter_trials = {}, {}, []
        for user_set in range(1, 9):
            folder_path_attention = (
                self.path
                + str(folder_attention)
                + "/"
                + self.model_name.split("/")[1]
                + "/set_"
                + str(user_set)
                + "/"
            )
            folder_fixations = (
                self.path + "gaze_features_real" + "/set_" + str(user_set) + "/"
            )
            # load fixations of user and concat
            real_gaze_fixations_user = HumanAttentionExtractor().load_gaze_features(
                folder_fixations
            )
            real_gaze_fixations = {**real_gaze_fixations_user, **real_gaze_fixations}
            # load attention of user and concat
            print(folder_path_attention)
            attention_trials_user = ModelAttentionExtractor.load_attention_df(
                folder_path_attention
            )
            attention_trials = {**attention_trials, **attention_trials_user}

            # load filter triasl info
            info_trials = HumanAttentionExtractor().load_trials_info(folder_fixations)
            # filter for all trial that were answered with the 3 particpants of this session
            if filter_completed is True:
                filter_trials_user = info_trials["complete"]
            else:
                filter_trials_user = info_trials["all"]
            # -------------------------------------------
            # filter for the chosen and the rejected ones
            if filter_cr == "chosen":
                # remove all trials dont end with .1
                filter_trials_user = [
                    trial for trial in filter_trials_user if str(trial).endswith(".1")
                ]
            elif filter_cr == "rejected":
                # remove all trials dont end with .1
                filter_trials_user = [
                    trial
                    for trial in filter_trials_user
                    if not str(trial).endswith(".1")
                ]
            filter_trials.extend(filter_trials_user)
            # -------------------------------------------

        sc_layers, sc_layers_all = {}, {}
        for layer in list(list(attention_trials.values())[0].keys()):
            mean_spearman, std_spearman, all_spearman = self.compute_sc_layer(
                real_gaze_fixations,
                attention_trials,
                gaze_feature=gaze_feature,
                layer=layer,
                filter_trials=filter_trials,
            )
            sc_layers[layer] = [mean_spearman, std_spearman]
            sc_layers_all[layer] = all_spearman

        return sc_layers, sc_layers_all

    def compute_sc_user_set(
        self,
        user_set,
        gaze_feature="fix_duration_n",
        filter_completed=False,
        filter_cr=False,
        folder_attention="attention",
    ):
        # Load the model
        folder_path_attention = (
            self.path
            + str(folder_attention)
            + "/"
            + self.model_name.split("/")[1]
            + "/set_"
            + str(user_set)
            + "/"
        )
        folder_fixations = (
            self.path + "gaze_features_real" + "/set_" + str(user_set) + "/"
        )
        real_gaze_fixations = HumanAttentionExtractor().load_gaze_features(
            folder_fixations
        )
        info_trials = HumanAttentionExtractor().load_trials_info(folder_fixations)
        # filter for all trial that were answered with the 3 particpants of this session
        if filter_completed is True:
            filter_trials = info_trials["complete"]
        else:
            filter_trials = info_trials["all"]
        # -------------------------------------------
        # filter for the chosen and the rejected ones
        if filter_cr == "chosen":
            # remove all trials dont end with .1
            filter_trials = [
                trial for trial in filter_trials if str(trial).endswith(".1")
            ]
        elif filter_cr == "rejected":
            # remove all trials dont end with .2
            filter_trials = [
                trial for trial in filter_trials if not str(trial).endswith(".1")
            ]
        # -------------------------------------------

        attention_trials = ModelAttentionExtractor.load_attention_df(
            folder_path_attention
        )
        sc_layers = {}
        for layer in list(list(attention_trials.values())[0].keys()):
            mean_spearman, std_spearman, all_spearman = self.compute_sc_layer(
                real_gaze_fixations,
                attention_trials,
                gaze_feature=gaze_feature,
                layer=layer,
                filter_trials=filter_trials,
            )
            sc_layers[layer] = mean_spearman

        return sc_layers

    def plot_attention_all_trials(
        self, gaze_features: list, attention_folder="attention", filter_completed=True
    ):
        folder_filter = "completed" if filter_completed else "not_filtered"
        path = (
            str(self.path)
            + attention_folder
            + "/"
            + str(self.model_name.split("/")[1])
            + str("/")
            + str(folder_filter)
        )
        data = pd.DataFrame()
        for gaze_feature in gaze_features:
            gaze_feature_data = pd.read_csv(
                path + "/correlation_trials_" + str(gaze_feature) + ".csv",
                sep=";",
                index_col=0,
            )
            gaze_feature_data = gaze_feature_data[["mean"]]
            gaze_feature_data.rename(columns={"mean": gaze_feature}, inplace=True)
            data = pd.concat([data, gaze_feature_data], axis=1)
        attention_matrix = data.values
        features_name = [self.gaze_features_names[x] for x in data.columns.to_list()]
        n_features = data.shape[1]
        # Create a heatmap
        n_layer = data.shape[0]
        colors = [
            "#E6F2FB",
            "#D2ECF5",
            "#BBDBEC",
            "#A4C8E1",
            "#A8B3E1",
            "#91A6D8",
            "#7A99CE",
            "#638CC5",
            "#4C7FBB",
        ]

        plt.figure(figsize=(5, 8))
        # Define a gradient scale with more shades of blue
        # -----------------------------------------
        blue_scale_colors = [
            "#e0f2f9",  # Very light blue
            "#8cc5e3",  # Lighter blue
            "#1a80bb",  # Original blue
            "#0a4f7d",  # Darker blue
            # "#002147"   # Very dark blue (navy)
        ]

        # Create a custom colormap with the blue gradient
        blue_cmap = LinearSegmentedColormap.from_list(
            "custom_blue_scale", blue_scale_colors
        )
        heatmap = sns.heatmap(
            attention_matrix,
            cmap=blue_cmap,
            annot=False,
            cbar=True,
            linewidths=0.3,
            linecolor="white",
        )
        # -----------------------------------------

        # heatmap = sns.heatmap(attention_matrix, cmap="viridis", annot=False, cbar=True)
        plt.title(str(self.model_name.split("/")[1]), fontsize=20, weight="regular")
        # plt.xlabel(str(self.model_name.split("/")[1]), fontsize=18, weight="bold")  # Increase x-axis label size
        plt.ylabel(
            "Layer Number",
            fontsize=18,  # weight="bold"
        )  # Increase y-axis label size
        plt.yticks(
            ticks=np.arange(0.5, n_layer, 1),
            labels=[f"L{n_layer-i}" for i in range(n_layer)],
            rotation=0,
            fontsize=16,
        )  # Increase y-tick font size
        plt.xticks(
            ticks=np.arange(0.5, n_features, 1),
            labels=features_name,
            rotation=0,
            fontsize=16,
        )  # Increase x-tick font size
        colorbar = heatmap.collections[0].colorbar
        colorbar.ax.tick_params(labelsize=16)
        colorbar.set_label("Correlation", fontsize=13)
        # # Add dashed lines (optional for dividing sections)
        # plt.axhline(4, color='orange', linestyle='--', lw=2)
        # plt.axhline(8, color='orange', linestyle='--', lw=2)
        # plt.axhline(12, color='orange', linestyle='--', lw=2)

        # Show the plot
        plt.tight_layout()
        path = (
            str(self.path)
            + attention_folder
            + "/"
            + "results"
            + str("/")
            + str("plots/attention_layers/")
        )
        # check if path exists
        if not os.path.exists(path):
            os.makedirs(path)
        plt.savefig(
            path
            + "/"
            + str(self.model_name.split("/")[1])
            + "_attention_layers_all_trials.png"
        )  # You can specify the format, e.g., .png, .pdf, .svg, etc.
        plt.show()
