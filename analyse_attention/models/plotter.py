import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
import os


class Plotter:
    @staticmethod
    def plot_gaze_signal(path, df, gaze_signal, tag="", plot_std=True):
        # Replace with your actual standard deviations
        categories = df.columns.tolist()
        try:
            means = [df.loc["mean", model] for model in df.columns]
            stds = [df.loc["std", model] for model in df.columns]
        except KeyError:
            print("error")
        data = list(zip(categories, means, stds))

        # Sort the data by means (ascending order)
        sorted_data = sorted(data, key=lambda x: x[1])  # Sort by mean value
        sorted_categories, sorted_means, sorted_stds = zip(
            *sorted_data
        )  # Unzip sorted data

        # Define elegant colors for each bar
        # elegant_colors = [
        #     "#4C7FBB",
        #     "#A4C8E1",
        #     "#FFC107",
        #     "#FF6F61",
        #     "#6B5B95",
        #     "#88B04B",
        #     "#F7CAC9",
        #     "#92B558",
        #     "#955251",
        #     "#FFB7B2",
        #     "#F2BB66",
        #     "#C49A6A",
        # ]
        elegant_colors = [
            "#4C7FBB",  # Original Blue
            "#638CC5",  # Lighter Blue
            "#7A99CE",  # Lighter Blue
            "#91A6D8",  # Lighter Blue
            "#A8B3E1",  # Lighter Blue
            "#A4C8E1",  # Original Light Blue
            "#BBDBEC",  # Even lighter Blue
            "#D2ECF5",  # Almost White Blue
            "#E6F2FB",  # Very Pale Blue
            "#FFC107",  # Original Yellow
            "#FFDC6B",  # Lighter Yellow
            "#FFE08F",  # Pale Yellow
            "#FF6F61",  # Original Coral
            "#FF9488",  # Lighter Coral
            "#FFB1A4",  # Lighter Coral
            "#F7CAC9",  # Soft Pink
            "#F2D8D7",  # Lighter Pink
            "#C49A6A",  # Elegant Brown
            "#D4A97A",  # Lighter Brown
            "#E2B990",  # Soft Beige
        ]
        elegant_colors = elegant_colors[: len(categories)]
        # Bar plot
        x = np.arange(len(sorted_means))  # The label locations
        width = 0.4  # The width of the bars

        fig, ax = plt.subplots(figsize=(10, 6))  # Set the figure size

        # Create the bar plot with elegant colors
        bars = ax.bar(x, sorted_means, width, color=elegant_colors, edgecolor="black")

        # Add shaded area for standard deviations
        if plot_std:
            for i in range(len(sorted_means)):
                ax.fill_between(
                    [x[i] - width / 2, x[i] + width / 2],
                    sorted_means[i] - sorted_stds[i],
                    sorted_means[i] + sorted_stds[i],
                    color="lightgray",
                    alpha=0.5,
                )

        # Adding labels and title
        ax.set_xlabel("Models", fontsize=12)
        ax.set_ylabel("Correlation", fontsize=12)
        # ax.set_title('', fontsize=14, fontweight='bold')
        ax.set_xticks(x)
        ax.set_xticklabels(
            sorted_categories, rotation=45, ha="right", fontsize=10
        )  # Vertical labels with right alignment

        # Adding grid for better readability
        ax.yaxis.grid(True, linestyle="--", alpha=0.7)

        # Show the plot
        plt.tight_layout()
        plt.savefig(
            path + "plots/" + tag + "_" + gaze_signal + "_barplot_means_stds.png"
        )
        plt.show()

    @staticmethod
    def plot_gaze_signal_chosenrejected(
        path, df, gaze_signal, tag="", plot_std=True, p_values={}
    ):
        feature_name = gaze_signal.split("_")[0]

        categories = df["chosen"].columns.tolist()
        palette_colors = [
            "#8cc5e3",  # Original Blue
            "#1a80bb",  # Lighter Blue
        ]

        means_df1 = [df["chosen"].loc["mean", model] for model in df["chosen"].columns]
        stds_df1 = [df["chosen"].loc["std", model] for model in df["chosen"].columns]
        means_df2 = [
            df["rejected"].loc["mean", model] for model in df["rejected"].columns
        ]
        stds_df2 = [
            df["rejected"].loc["std", model] for model in df["rejected"].columns
        ]
        data1 = list(zip(categories, means_df1, stds_df1))
        data2 = list(zip(categories, means_df2, stds_df2))

        # Sort the data by means (ascending order)
        sorted_data = sorted(data1, key=lambda x: x[1])  # Sort by mean value
        sorted_categories, means_df1, stds_df1 = zip(*sorted_data)  # Unzip sorted data
        sorted_data = sorted(data2, key=lambda x: sorted_categories.index(x[0]))
        categories, means_df2, stds_df2 = zip(*sorted_data)  # Unzip sorted data
        p_values_ordered = {k: p_values[k] for k in list(sorted_categories)}
        x = np.arange(len(categories))  # The label locations
        width = 0.35  # The width of the bars
        fig, ax = plt.subplots(figsize=(12, 6))  # Set the figure size
        # Create the bar plots for both dataframes
        bars_df1 = ax.bar(
            x - width / 2,
            means_df1,
            width,
            color=palette_colors[0],
            edgecolor="black",
            linewidth=1,
            label="Chosen",
            yerr=stds_df1 if plot_std else None,
            capsize=5,
        )
        bars_df2 = ax.bar(
            x + width / 2,
            means_df2,
            width,
            color=palette_colors[1],
            edgecolor="black",
            linewidth=1,
            label="Rejected",
            yerr=stds_df2 if plot_std else None,
            capsize=5,
        )
        for i, category in enumerate(categories):
            p_value = p_values_ordered.get(
                category, 1
            )  # Default to 1 if category is not in the dictionary

            if p_value < 0.001:
                # Determine the height for the star (above the taller bar)
                y_max = max(
                    means_df1[i] + (stds_df1[i] if plot_std else 0),
                    means_df2[i] + (stds_df2[i] if plot_std else 0),
                )
                ax.text(
                    x[i],  # x position (center between the two bars)
                    y_max + 0.0006,  # y position slightly above the bar
                    "***",  # The text to display
                    ha="center",
                    va="bottom",
                    color="black",
                    fontsize=8,
                    fontweight="bold",
                )
            elif p_value < 0.01:
                # Determine the height for the star (above the taller bar)
                y_max = max(
                    means_df1[i] + (stds_df1[i] if plot_std else 0),
                    means_df2[i] + (stds_df2[i] if plot_std else 0),
                )
                ax.text(
                    x[i],  # x position (center between the two bars)
                    y_max + 0.0006,  # y position slightly above the bar
                    "**",  # The text to display
                    ha="center",
                    va="bottom",
                    color="black",
                    fontsize=8,
                    fontweight="bold",
                )
            elif p_value < 0.05:
                # Determine the height for the star (above the taller bar)
                y_max = max(
                    means_df1[i] + (stds_df1[i] if plot_std else 0),
                    means_df2[i] + (stds_df2[i] if plot_std else 0),
                )
                ax.text(
                    x[i],  # x position (center between the two bars)
                    y_max + 0.0006,  # y position slightly above the bar
                    "*",  # The text to display
                    ha="center",
                    va="bottom",
                    color="black",
                    fontsize=8,
                    fontweight="bold",
                )

        # Adding labels and title
        # ax.set_xlabel("Models", fontsize=12)
        ax.set_ylabel("Correlation", fontsize=16)
        ax.set_title(
            "Model attention outputs correlation with " + feature_name,
            fontsize=18,
            # fontweight="bold",
        )
        ax.set_xticks(x)
        ax.set_xticklabels(categories, rotation=90, ha="right", fontsize=15)
        ax.tick_params(axis="y", labelsize=16)
        ax.legend(fontsize=14)  # Set legend font size
        ax.yaxis.grid(True, linestyle="--", alpha=0.7)

        # Show the plot
        plt.tight_layout()
        path_to_save = path + "plots"
        if not os.path.exists(path_to_save):
            os.makedirs(path_to_save)
        plt.savefig(
            path_to_save
            + "/"
            + tag
            + "_"
            + gaze_signal
            + "_barplot_means_stds_chosenrejected.png"
        )
        plt.show()

    @staticmethod
    def plot_gaze_signal_chosenrejected_reward(
        path,
        df1,
        df2,
        gaze_signal,
        tag="",
        plot_std=False,
        p_values_df1={},
        p_values_df2={},
    ):
        palette_colors = [
            "#8cc5e3",  # Original Blue
            "#1a80bb",  # Lighter Blue
        ]
        feature_name = gaze_signal.split("_")[0]
        df_chosen = Plotter.concatenate_reward_dfs(df1["chosen"], df2["chosen"])
        df_rejected = Plotter.concatenate_reward_dfs(df1["rejected"], df2["rejected"])
        categories = df_chosen.columns.tolist()
        # elegant_colors = elegant_colors[: len(categories)]
        means_df1 = [df_chosen.loc["mean", model] for model in df_chosen.columns]
        stds_df1 = [df_chosen.loc["std", model] for model in df_chosen.columns]
        means_df2 = [df_rejected.loc["mean", model] for model in df_rejected.columns]
        stds_df2 = [df_rejected.loc["std", model] for model in df_rejected.columns]
        data1 = list(zip(categories, means_df1, stds_df1))
        data2 = list(zip(categories, means_df2, stds_df2))
        # -------------------------------------------------------------------
        # Sort the data by means (ascending order)
        # sorted_data = sorted(data1, key=lambda x: x[1])  # Sort by mean value
        # Sort the alphabetic order
        sorted_data = sorted(
            data1, key=lambda x: x[0], reverse=True
        )  # Sort by mean value
        sorted_categories, means_df1, stds_df1 = zip(*sorted_data)  # Unzip sorted data
        sorted_data = sorted(data2, key=lambda x: sorted_categories.index(x[0]))
        categories, means_df2, stds_df2 = zip(*sorted_data)  # Unzip sorted data
        # -------------------------------------------------------------------
        p_values_ordered = {
            k: p_values_df1[k]
            if not str(k).endswith("_r")
            else p_values_df2[k.split("_r")[0]]
            for k in list(categories)
        }

        x = np.arange(len(categories))  # The label locations
        width = 0.35  # The width of the bars

        fig, ax = plt.subplots(figsize=(9, 6))  # Set the figure size

        # Create the bar plots for both dataframes
        bars_df1 = ax.bar(
            x - width / 2,
            means_df1,
            width,
            color=palette_colors[0],
            edgecolor="black",
            linewidth=1,
            label="Chosen",
            yerr=stds_df1 if plot_std else None,
            capsize=5,
        )
        bars_df2 = ax.bar(
            x + width / 2,
            means_df2,
            width,
            color=palette_colors[1],
            edgecolor="black",
            linewidth=1,
            label="Rejected",
            yerr=stds_df2 if plot_std else None,
            capsize=5,
        )

        for i, category in enumerate(categories):
            p_value = p_values_ordered.get(
                category, 1
            )  # Default to 1 if category is not in the dictionary

            if p_value < 0.001:
                # Determine the height for the star (above the taller bar)
                y_max = max(
                    means_df1[i] + (stds_df1[i] if plot_std else 0),
                    means_df2[i] + (stds_df2[i] if plot_std else 0),
                )
                ax.text(
                    x[i],  # x position (center between the two bars)
                    y_max + 0.0006,  # y position slightly above the bar
                    "***",  # The text to display
                    ha="center",
                    va="bottom",
                    color="black",
                    fontsize=8,
                    fontweight="bold",
                )
            elif p_value < 0.01:
                # Determine the height for the star (above the taller bar)
                y_max = max(
                    means_df1[i] + (stds_df1[i] if plot_std else 0),
                    means_df2[i] + (stds_df2[i] if plot_std else 0),
                )
                ax.text(
                    x[i],  # x position (center between the two bars)
                    y_max + 0.0006,  # y position slightly above the bar
                    "**",  # The text to display
                    ha="center",
                    va="bottom",
                    color="black",
                    fontsize=8,
                    fontweight="bold",
                )
            elif p_value < 0.05:
                # Determine the height for the star (above the taller bar)
                y_max = max(
                    means_df1[i] + (stds_df1[i] if plot_std else 0),
                    means_df2[i] + (stds_df2[i] if plot_std else 0),
                )
                ax.text(
                    x[i],  # x position (center between the two bars)
                    y_max + 0.0006,  # y position slightly above the bar
                    "*",  # The text to display
                    ha="center",
                    va="bottom",
                    color="black",
                    fontsize=8,
                    fontweight="bold",
                )

        # Adding labels and title
        # ax.set_xlabel("Models", fontsize=12)
        ax.set_ylabel("Correlation", fontsize=16)
        ax.set_title(
            "Model attention outputs correlated with " + feature_name,
            fontsize=16,
            # fontweight="bold",
        )
        # ax.set_title('Grouped Bar Plot with Means and Standard Deviations', fontsize=14, fontweight='bold')
        ax.set_xticks(x)
        ax.set_xticklabels(
            categories, rotation=90, ha="right", fontsize=16
        )  # Vertical labels with right alignment
        ax.legend(loc="upper center", fontsize=12, framealpha=0)
        ax.yaxis.grid(True, linestyle="--", alpha=0.7)

        # Show the plot
        plt.tight_layout()
        path_to_save = path + "plots"
        if not os.path.exists(path_to_save):
            os.makedirs(path_to_save)
        plt.savefig(
            path_to_save
            + "/"
            + tag
            + "_"
            + gaze_signal
            + "_barplot_means_stds_chosenrejected_reward.png"
        )
        plt.show()

    @staticmethod
    def concatenate_reward_dfs(df1, df2):
        # Step 2: Keep only the common columns in both DataFrames
        common_columns = df1.columns.intersection(df2.columns)
        df1_common = df1[common_columns]
        df2_common = df2[common_columns]

        # Step 3: Rename columns of df2 by appending "_r"
        df2_common = df2_common.rename(
            columns={col: col + "_r" for col in common_columns}
        )

        # Step 4: Concatenate the DataFrames horizontally
        return pd.concat([df1_common, df2_common], axis=1)
