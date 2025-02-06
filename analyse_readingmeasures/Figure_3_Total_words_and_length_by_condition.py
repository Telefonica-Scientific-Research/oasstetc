# -*- coding: utf-8 -*-


#%% Import packages
import pathlib
import os, glob, warnings
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.stats import ttest_ind
from scipy.stats import ttest_rel
warnings.filterwarnings('ignore')
plt.rcParams.update({'font.size': 16})

parent_path = str(pathlib.Path(__file__).parent.resolve().parent.resolve())


base_path=  parent_path + "/oasstetc_data/raw_data/"


palette_colors = [
    "#8cc5e3",  # Original Blue
    "#1a80bb",  # Lighter Blue
]

# METRICS TO CALCULATE
metrics = [
    {
        "name": "Total Words", "y_label": "Total Words",
        "function": lambda df: df.shape[0]
    },
    {
        "name": "Average Word Length", "y_label": "Average Word Length",
        "function": lambda df: df['text'].apply(lambda x: len(str(x))).mean()
    }
]

def process_n_prompt(n_prompt):
    if len(n_prompt) > 1 and n_prompt[-2] == '.':
        return n_prompt[:-2]
    else:
        return n_prompt

results = {}
participants = list(range(1, 25)) 
idx_dataset_used = [1,2,3,4,5,6,7,8,5,5,6,1,1,6,8,8,3,3,2,2,4,4,7,7]

all_metric_values = {metric['name']: [] for metric in metrics}
group_data = {metric['name']: {"Preferred": [], "Non-Preferred": []} for metric in metrics}

for participant_number in participants:
    dataset_used=idx_dataset_used[participant_number-1]
    participant_folder = os.path.join(base_path, f'participant_{participant_number}_{dataset_used}')
    files = glob.glob(os.path.join(participant_folder, 'session_1\\vertices\\*word_cor*.csv'))
    results[participant_number] = {"preferred": {metric['name']: [] for metric in metrics},
                                   "non-preferred": {metric['name']: [] for metric in metrics}}
    test_file_path = glob.glob(os.path.join(participant_folder, 'session_1\\', '*TEST*.csv'))[0]
    df_test = pd.read_csv(test_file_path, delimiter=';;;')
    file_codes = [os.path.basename(f).split('_')[-1].replace('.csv', '')[:-2] for f in files]
    df_test['n_prompt'] = df_test['n_prompt'].astype(str)
    df_test['n_prompt_processed'] = df_test['n_prompt'].apply(process_n_prompt)
    df_test_filtered = df_test[df_test['n_prompt_processed'].isin(file_codes)]
    df_test_filtered = df_test_filtered.drop(df_test_filtered.index[2::3]).reset_index(drop=True)   
    df_test_filtered = df_test_filtered[["n_resp"]]

    idx_temp = 0
    for file in files:
        if file.endswith('.1.csv'):
            condition = "preferred"
        else:
            condition = "non-preferred"
        
        df = pd.read_csv(file, delimiter=';')
        if "fixations" not in df.columns:
            continue
        columns_to_keep = ["number", "text", "fixations", "fix_duration", "fix_number", 
                           "fix_duration_avg", "pupil", "first_fix_duration"]
        df_filtered = df[columns_to_keep]
        
        for metric in metrics:
            metric_name = metric['name']
            metric_value = metric['function'](df_filtered)
            results[participant_number][condition][metric_name].append(metric_value)
            all_metric_values[metric_name].append(metric_value)  # Collect all metric data

    for metric in metrics:
        metric_name = metric['name']
        preferred_values = results[participant_number]["preferred"][metric_name]
        non_preferred_values = results[participant_number]["non-preferred"][metric_name]
        avg_preferred = sum(preferred_values) / len(preferred_values) if preferred_values else 0
        avg_non_preferred = sum(non_preferred_values) / len(non_preferred_values) if non_preferred_values else 0
        group_data[metric_name]["Preferred"].append(avg_preferred)
        group_data[metric_name]["Non-Preferred"].append(avg_non_preferred)
#%%
from scipy.stats import ttest_rel
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

def generate_group_comparison_plots(exclude_participant=None):
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))  # Adjust the height by changing the second value
    colors = ['lightblue', 'lightgreen']  # Colors for the two conditions

    for idx, metric in enumerate(metrics):
        metric_name = metric['name']
        y_label = metric['y_label']
        group_preferred = [val for i, val in enumerate(group_data[metric_name]["Preferred"]) if participants[i] != exclude_participant]
        group_non_preferred = [val for i, val in enumerate(group_data[metric_name]["Non-Preferred"]) if participants[i] != exclude_participant]
        
        group_df_plot = pd.DataFrame({
            "Condition": ["Chosen"] * len(group_preferred) + ["Rejected"] * len(group_non_preferred),
            y_label: group_preferred + group_non_preferred
        })
        
        if len(group_preferred) > 0 and len(group_non_preferred) > 0:
            # Perform paired t-test (ttest_rel)
            t_stat_preferred, p_value_preferred = ttest_rel(group_preferred, group_non_preferred)
            
            # Calculate Cohen's d
            differences = np.array(group_preferred) - np.array(group_non_preferred)
            cohen_d = np.mean(differences) / np.std(differences, ddof=1)
            
            # Calculate degrees of freedom
            df = len(group_preferred) - 1
            
            # Print the t-statistic, p-value, degrees of freedom, and Cohen's d to the console
            print(f"{metric_name} - t-statistic: {t_stat_preferred:.4f}, p-value: {p_value_preferred:.4f}, df: {df}, Cohen's d: {cohen_d:.4f}")
        else:
            t_stat_preferred, p_value_preferred, cohen_d, df = None, None, None, None
        
        # Create boxplot with distinctive colors for each condition within the same panel
        sns.boxplot(
            x='Condition', y=y_label, data=group_df_plot, palette=palette_colors, ax=axes[idx], showfliers=False,
            boxprops=dict(edgecolor='black'), whiskerprops=dict(color='black'), capprops=dict(color='black'),
            medianprops=dict(color='black')
        )
        
        # Get the current y-limits and adjust for the significance line and asterisks
        y_min, y_max = axes[idx].get_ylim()
        
        # Add a line with 3 asterisks above the whiskers and adjust their positions
        if p_value_preferred is not None and p_value_preferred < 0.05:
            x1, x2 = 0, 1   # positions for the boxplots
            
            if idx == 0:  # Left panel (Total Words)
                y = y_max * 1.01  # Slightly lower the line and asterisks for the left panel
                h = (y_max - y_min) * 0.015
            else:  # Right panel (Average Word Length)
                y = y_max * 1.02  # Keep the original position for the right panel
                h = (y_max - y_min) * 0.02
            
            axes[idx].plot([x1, x2], [y, y], lw=1.5, c='black')  # Line between the conditions
            axes[idx].text((x1 + x2) * 0.5, y + h, '***', ha='center', va='bottom', color='black')  # Asterisks placed right above the line

        # Adjust the y-limit slightly to accommodate the significance line
        axes[idx].set_ylim(y_min, y_max * 1.05)

        axes[idx].set_title(f'{metric_name}')
        axes[idx].set_xlabel('')

    plt.tight_layout()
    plt.show()

# Call the function
generate_group_comparison_plots(exclude_participant=None)
