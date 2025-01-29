# -*- coding: utf-8 -*-
"""
Created on Wed Oct 23 11:12:48 2024

@author: id05278
"""

#%% Import packages
import os, glob, warnings, json
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.stats import ttest_rel
import numpy as np
warnings.filterwarnings('ignore')
plt.rcParams.update({'font.size': 18})  # Adjusted font size

# Base path to the datasets - Modify to convenience
base_paths = {
    'real': r'C:\Users\id05278\Documents\GitHub\ETHAPT\gaze_features_real',
    'synthetic': r'C:\Users\id05278\Documents\GitHub\ETHAPT\gaze_features_synthetic'
}

# Metrics to accumulate for real and synthetic data
metrics_config = {
    'real': {
        'metrics': ["fix_duration", "first_fix_duration", "fix_number"],
        'columns_to_keep': ["fix_duration", "first_fix_duration", "fix_number", "text"]
    },
    'synthetic': {
        'metrics': ['fix_duration', 'first_fix_duration', 'fix_number'],
        'columns_to_keep': ['fix_duration', 'first_fix_duration', 'fix_number', 'text']
    }
}

# Color palette for the boxplots
palette_colors = {
    'Chosen': '#1a80bb',  # Lighter blue for "Chosen"
    'Rejected': '#8cc5e3'  # Original blue for "Rejected"
}

# Function to accumulate and calculate metrics per dataset
def accumulate_metrics(data_type, use_complete_trials):
    metrics = metrics_config[data_type]['metrics']
    columns_to_keep = metrics_config[data_type]['columns_to_keep']

    dataset_metrics = {}

    for dataset_idx in range(1, 9):
        dataset_folder = os.path.join(base_paths[data_type], f'set_{dataset_idx}')
        dataset_real = os.path.join(base_paths['real'], f'set_{dataset_idx}')

        json_file = os.path.join(dataset_real, 'info_trials.json')
        with open(json_file, 'r') as f:
            trials_info = json.load(f)

        selected_trials = trials_info["complete"] if use_complete_trials else trials_info["all"]
        selected_trials = [str(trial)[:-2] for trial in selected_trials]

        dataset_metrics[dataset_idx] = {}
        
        files = glob.glob(os.path.join(dataset_folder, 'word*.csv'))
        file_codes = [os.path.basename(f).split('_')[-1].replace('.csv', '')[:-2] for f in files]
        
        filtered_files = [f for f, code in zip(files, file_codes) if code in selected_trials]
        
        for file_path in filtered_files:
            file_code = os.path.basename(file_path).split('_')[-1].replace('.csv', '')[:-2]
            condition = "preferred" if file_path.endswith('.1.csv') else "non-preferred"
            
            if file_code not in dataset_metrics[dataset_idx]:
                dataset_metrics[dataset_idx][file_code] = {
                    "metrics": {"preferred": {metric: [] for metric in metrics}, 
                                "non-preferred": {metric: [] for metric in metrics}},
                    "users": 0
                }
            
            df = pd.read_csv(file_path, delimiter=';')
            
            if not set(columns_to_keep).issubset(df.columns):
                continue
            
            df_filtered = df[columns_to_keep]
            df_filtered['text_length'] = df_filtered['text'].str.len()
            df_filtered = df_filtered.dropna(subset=['text_length', 'fix_duration'])
            
            if df_filtered.empty:
                continue
            
            for metric in metrics:
                dataset_metrics[dataset_idx][file_code]["metrics"][condition][metric].append(df_filtered[metric].mean())
            
            dataset_metrics[dataset_idx][file_code]["users"] += 0.5

    return dataset_metrics

#%% Function to generate combined plots for real or synthetic data

def generate_combined_plots(dataset_metrics, metric, ax, title, is_all):
    combined_preferred = []
    combined_non_preferred = []
    
    for dataset in dataset_metrics:
        for file_code, data in dataset_metrics[dataset].items():
            preferred_values = data["metrics"]["preferred"].get(metric)
            non_preferred_values = data["metrics"]["non-preferred"].get(metric)
    
            if preferred_values is not None and non_preferred_values is not None:
                if len(preferred_values) > 0 and len(non_preferred_values) > 0:
                    combined_preferred.extend(preferred_values)
                    combined_non_preferred.extend(non_preferred_values)

    # Ensure both lists are of equal length before performing the t-test
    if len(combined_preferred) != len(combined_non_preferred):
        print(f"Skipping t-test for {title}: unequal lengths (Chosen: {len(combined_preferred)}, Rejected: {len(combined_non_preferred)})")
        return

    # Paired t-test
    t_stat_combined, p_value_combined = ttest_rel(combined_preferred, combined_non_preferred)
    degrees_of_freedom = len(combined_preferred) - 1
    
    # Calculate Cohen's d for effect size
    differences = np.array(combined_preferred) - np.array(combined_non_preferred)
    cohen_d = np.mean(differences) / np.std(differences, ddof=1)

    # Print the t-statistic, p-value, degrees of freedom, and Cohen's d to the console
    print(f"T-test results for {metric}: t-stat = {t_stat_combined:.4f}, p-value = {p_value_combined:.4f}, df = {degrees_of_freedom}, Cohen's d = {cohen_d:.4f}")

    # Create dataframe for plotting
    df_combined = pd.DataFrame({
        "Condition": ["Chosen"] * len(combined_preferred) + ["Rejected"] * len(combined_non_preferred),
        metric: combined_preferred + combined_non_preferred
    })

    # Generate boxplot with distinctive colors
    sns.boxplot(
        x='Condition', y=metric, data=df_combined, 
        palette=palette_colors, ax=ax, showfliers=False
    )

    # Remove x-axis label
    ax.set_xlabel('')

    # Add significance asterisks if the p-value is significant
    if p_value_combined < 0.05:
        asterisks = '**' if p_value_combined < 0.01 else '*'
        ax.annotate(asterisks, 
            xy=(0.5, np.percentile(df_combined[metric], 75) * 1.15),
            xycoords='data', ha='center', va='bottom', fontsize=18, color='black')

    # Modify y-axis label to include (All) or (UR)
    pretty_metric = {
        'fix_duration': 'TRT',
        'first_fix_duration': 'FFD',
        'fix_number': 'nFix'
    }.get(metric, metric.replace('_', ' ').title())

    if is_all:
        ax.set_ylabel(f'{pretty_metric} (All)')
    else:
        ax.set_ylabel(f'{pretty_metric} (UR)')


#%% Generate 1x6 figures for real and synthetic data
fig_real, axes_real = plt.subplots(1, 6, figsize=(24, 6))  # Adjusted figure width

# Real data (combining all in a single row)
for idx, use_complete_trials in enumerate([False, True]):
    dataset_metrics_real = accumulate_metrics('real', use_complete_trials)
    for jdx, metric in enumerate(metrics_config['real']['metrics']):
        position = idx * 3 + jdx
        is_all = idx == 0  # First 3 panels are "All", second 3 are "UR"
        generate_combined_plots(dataset_metrics_real, metric, axes_real[position], metric.replace('_', ' ').title(), is_all)

fig_real.suptitle('Organic Data')  # Simplified title
plt.tight_layout()
plt.show()

fig_synthetic, axes_synthetic = plt.subplots(1, 6, figsize=(24, 6))  # Adjusted figure width

# Synthetic data (combining all in a single row)
for idx, use_complete_trials in enumerate([False, True]):
    dataset_metrics_synthetic = accumulate_metrics('synthetic', use_complete_trials)
    for jdx, metric in enumerate(metrics_config['synthetic']['metrics']):
        position = idx * 3 + jdx
        is_all = idx == 0  # First 3 panels are "All", second 3 are "UR"
        generate_combined_plots(dataset_metrics_synthetic, metric, axes_synthetic[position], metric.replace('_', ' ').title(), is_all)

fig_synthetic.suptitle('Synthetic Data')  # Simplified title
plt.tight_layout()
plt.show()

