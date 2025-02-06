#%% Import packages
import pathlib
import os, glob, warnings, mne
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.stats import ttest_rel, ttest_ind
import numpy as np
from datetime import datetime, timezone, timedelta

mne.set_log_level("WARNING")
warnings.filterwarnings('ignore')
parent_path = str(pathlib.Path(__file__).parent.resolve().parent.resolve())


base_path=  parent_path + "/oasstetc_data/raw_data/"

# Metrics: Mean, std, max, min, median and range
metrics = [
    {"name": "Mean", "y_label": "Mean", "function": lambda df: df.mean(skipna=True)},
    {"name": "Standard Deviation", "y_label": "Standard Deviation", "function": lambda df: df.std(skipna=True)},
    {"name": "Minimum", "y_label": "Minimum", "function": lambda df: df.min(skipna=True)},
    {"name": "Maximum", "y_label": "Maximum", "function": lambda df: df.max(skipna=True)},
    {"name": "Median", "y_label": "Median", "function": lambda df: df.median(skipna=True)},
    {"name": "Range", "y_label": "Range", "function": lambda df: df.max(skipna=True) - df.min(skipna=True)},
]

results = {}
participants = list(range(1, 24))
signal_types = ["GSR", "ECG"]  # Add the signal types

# Mapping of equivalent signal types
signal_type_mapping = {
    "GSR": ["GSR", "EDA"],  # Treat GSR and EDA as equivalent
    "ECG": ["ECG", "PPG"]   # Treat ECG and PPG as equivalent
}

all_metric_values = {metric['name']: [] for metric in metrics}
group_data = {metric['name']: {"Preferred": [], "Non-Preferred": []} for metric in metrics}
data_by_participant = {signal: {} for signal in signal_types}  # Separate data by signal type
synchronize_starting_point = 1

for participant_number in participants:
    if synchronize_starting_point == 1:
        # EEG DATA TO SYNCHRONIZE
        participant_folder_EEG = os.path.join(base_path, f'participant_{participant_number}_*\\session_1')
        #if participant_number == 2:
        #    participant_folder_EEG = os.path.join(base_path, f'participant_{participant_number}\\session_2')
        hdr_files = list(glob.glob(os.path.join(participant_folder_EEG, '*.vhdr')))
        raw_hdr = mne.io.read_raw_brainvision(hdr_files[0], verbose=False, misc=[], scale=1.0); raw_hdr.load_data()
        edf_files = list(glob.glob(os.path.join(participant_folder_EEG, '*.edf')))
        
        if edf_files:
            # If EDF file is available, use it to determine the start time
            raw_edf = mne.io.read_raw_edf(edf_files[0], preload=True, verbose=False)
            acquisition_end_datetime = raw_edf.info['meas_date']
            print("Ending time:", acquisition_end_datetime)
            
            sfreq = raw_edf.info['sfreq']
            n_samples = raw_edf.n_times
            total_duration_seconds = n_samples / sfreq
            
            start_time = acquisition_end_datetime - timedelta(seconds=total_duration_seconds)
            print("Calculated EEG starting time:", start_time)
            start_time_EEG = start_time
        
        else:
            # If no EDF file, extract the time from the VHDR file name
            vhdr_filename = os.path.basename(hdr_files[0])
            # Example filename: [2024.09.18-10.16.21]_EEG.vhdr
            try:
                # Extract the datetime string from the filename using pattern matching
                date_time_str = vhdr_filename.split('_')[0].strip('[]')  # "2024.09.18-10.16.21"
                start_time_EEG = datetime.strptime(date_time_str, "%Y.%m.%d-%H.%M.%S")
                start_time_EEG = start_time_EEG.replace(tzinfo=timezone.utc)
                print("Extracted EEG starting time from filename:", start_time_EEG)
            except ValueError as e:
                print(f"Failed to extract start time from file name: {vhdr_filename}. Error: {e}")
                start_time_EEG = None
        
        # Eye tracker DATA TO SYNCHRONIZE
        files_eye_tracker = glob.glob(os.path.join(base_path, f'participant_{participant_number}_*\\session_1\\*TEST*.csv'))
        test_file_path = files_eye_tracker[0]; df_test = pd.read_csv(test_file_path, delimiter=';;;')
        print("Calculated ET starting time:", df_test.date[0]); start_time_ET_str = df_test.date[0]
    
        start_time_ET = datetime.strptime(start_time_ET_str, '%Y-%m-%d_%Hh%M.%S.%f'); start_time_ET = start_time_ET.replace(tzinfo=timezone.utc)
    
        time_difference = start_time_ET - start_time_EEG
        time_difference_seconds = abs(time_difference.total_seconds())
        print("Time difference in seconds:", time_difference_seconds)
    
    # Process signals, accounting for equivalent types
    for base_signal_type, equivalent_types in signal_type_mapping.items():
        participant_folder = os.path.join(base_path, f'participant_{participant_number}_*')
        
        # Search for any file matching the equivalent signal types
        file = []
        for signal_type in equivalent_types:
            file = glob.glob(os.path.join(participant_folder, f'session_1\\*{signal_type}*.csv'))
            if file:
                break  # Stop if a file is found

        if not file:
            continue  # Skip if no file found for any of the equivalent types
        
        df_signal = pd.read_csv(file[0], delimiter=',')
        
        if synchronize_starting_point == 1:
            df_signal = df_signal[df_signal['Time:256Hz'] >= time_difference_seconds].reset_index(drop=True)
            df_signal['Time:256Hz'] = df_signal['Time:256Hz'] - df_signal['Time:256Hz'].iloc[0]
        
        df_signal_downsampled = df_signal.iloc[::8, :].reset_index(drop=True)
        
        # Load eye-tracker file
        eyetracker_file_path = os.path.join(base_path, f'participant_{participant_number}_*\\session_1', '*TEST*.csv')
        eyetracker_files = glob.glob(eyetracker_file_path)
        df_test = pd.read_csv(eyetracker_files[0], delimiter=';;;')
        df_signal_downsampled['n_prompt_info'] = np.nan

        for idx, row in df_test.iterrows():
            if pd.notna(row['key_resp_2.started']) and pd.notna(row['question_useful_2.started']) and pd.notna(row['n_resp']):
                # Convert to float
                stimuli_start_time = float(row['key_resp_2.started'])
                stimuli_end_time = float(row['question_useful_2.started'])
                n_resp_value = row['n_resp']
                
                closest_start_idx = (np.abs(df_signal_downsampled['Time:256Hz'] - stimuli_start_time)).idxmin()
                closest_end_idx = (np.abs(df_signal_downsampled['Time:256Hz'] - stimuli_end_time)).idxmin()
                
                df_signal_downsampled.loc[closest_start_idx, 'n_prompt_info'] = f"{n_resp_value}S"
                df_signal_downsampled.loc[closest_end_idx, 'n_prompt_info'] = f"{n_resp_value}E"

        start_indices = df_signal_downsampled[df_signal_downsampled['n_prompt_info'].str.contains('S', na=False)].index
        end_indices = df_signal_downsampled[df_signal_downsampled['n_prompt_info'].str.contains('E', na=False)].index
        mask = np.zeros(len(df_signal_downsampled), dtype=bool)

        for start_idx, end_idx in zip(start_indices, end_indices):
            if start_idx < end_idx:
                mask[start_idx:end_idx + 1] = True
        
        df_signal_downsampled_filtered = df_signal_downsampled[mask].reset_index(drop=True)

        preferred_data = []
        non_preferred_data = []
        current_stimulus_data = []
        is_preferred = False
        collecting = False
        
        for idx, row in df_signal_downsampled_filtered.iterrows():
            if pd.notna(row['n_prompt_info']):
                if 'S' in row['n_prompt_info']:
                    collecting = True  
                    current_stimulus_data = []  
                    n_resp_value = row['n_prompt_info'].replace('S', '') 
                    is_preferred = '.1' in n_resp_value  

                if 'E' in row['n_prompt_info'] and collecting:
                    collecting = False  
                    if is_preferred:
                        preferred_data.append(current_stimulus_data)
                    else:
                        non_preferred_data.append(current_stimulus_data)
                    current_stimulus_data = []
            if collecting:
                current_stimulus_data.append(row['Channel 1'])

        if participant_number not in data_by_participant[base_signal_type]:
            data_by_participant[base_signal_type][participant_number] = {"preferred": [], "non-preferred": []}
        
        data_by_participant[base_signal_type][participant_number]["preferred"] = pd.DataFrame(preferred_data)
        data_by_participant[base_signal_type][participant_number]["non-preferred"] = pd.DataFrame(non_preferred_data)

    

#%%
for signal_type in signal_types:
    for participant_number in participants:
        preferred_df = data_by_participant[signal_type].get(participant_number, {}).get("preferred")
        non_preferred_df = data_by_participant[signal_type].get(participant_number, {}).get("non-preferred")

        if preferred_df is None or non_preferred_df is None:
            continue

        if participant_number not in results:
            results[participant_number] = {}

        if signal_type not in results[participant_number]:
            results[participant_number][signal_type] = {"preferred": {metric['name']: [] for metric in metrics},
                                                        "non-preferred": {metric['name']: [] for metric in metrics}}

        for metric in metrics:
            metric_name = metric['name']
            preferred_values = preferred_df.apply(metric['function'], axis=1)  # Apply row-wise
            non_preferred_values = non_preferred_df.apply(metric['function'], axis=1)  # Apply row-wise
            preferred_values = preferred_values.tolist()
            non_preferred_values = non_preferred_values.tolist()

            results[participant_number][signal_type]["preferred"][metric_name].extend(preferred_values)
            results[participant_number][signal_type]["non-preferred"][metric_name].extend(non_preferred_values)

            all_metric_values[metric_name].extend(preferred_values + non_preferred_values)
            group_data[metric_name]["Preferred"].extend(preferred_values)
            group_data[metric_name]["Non-Preferred"].extend(non_preferred_values)

#%%
def exclude_outliers(data):
    """
    Exclude outliers from the data based on the IQR method.
    """
    Q1 = np.percentile(data, 25)
    Q3 = np.percentile(data, 75)
    IQR = Q3 - Q1
    lower_bound = Q1 - 1.5 * IQR
    upper_bound = Q3 + 1.5 * IQR
    return [x for x in data if lower_bound <= x <= upper_bound]

#%% 
def calculate_group_stats(metric, signal_type):
    """
    Calculate mean and standard deviation for preferred and non-preferred data
    across participants for a given metric and signal type.
    """
    metric_name = metric['name']
    
    # Collect group data for preferred and non-preferred across all participants
    preferred_averages = []
    non_preferred_averages = []

    for participant_number in participants:
        if participant_number not in results or signal_type not in results[participant_number]:
            continue
        
        preferred_values = results[participant_number][signal_type]["preferred"][metric_name]
        non_preferred_values = results[participant_number][signal_type]["non-preferred"][metric_name]

        if len(preferred_values) > 0:
            preferred_avg = np.mean(preferred_values)
            preferred_averages.append(preferred_avg)
        
        if len(non_preferred_values) > 0:
            non_preferred_avg = np.mean(non_preferred_values)
            non_preferred_averages.append(non_preferred_avg)

    # Exclude outliers
    preferred_averages = exclude_outliers(preferred_averages)
    non_preferred_averages = exclude_outliers(non_preferred_averages)

    # Calculate mean and standard deviation
    preferred_mean = np.mean(preferred_averages) / 100000 if preferred_averages else np.nan
    preferred_std = np.std(preferred_averages) / 100000 if preferred_averages else np.nan
    non_preferred_mean = np.mean(non_preferred_averages) / 100000 if non_preferred_averages else np.nan
    non_preferred_std = np.std(non_preferred_averages) / 100000 if non_preferred_averages else np.nan

    return preferred_mean, preferred_std, non_preferred_mean, non_preferred_std

# Create empty DataFrames to store the results for EDA and PPG separately
eda_columns = ["Preferred EDA (Mean ± SD)", "Non-Preferred EDA (Mean ± SD)"]
ppg_columns = ["Preferred PPG (Mean ± SD)", "Non-Preferred PPG (Mean ± SD)"]
eda_table = pd.DataFrame(index=[metric["name"] for metric in metrics], columns=eda_columns)
ppg_table = pd.DataFrame(index=[metric["name"] for metric in metrics], columns=ppg_columns)

# Calculate the stats and fill the tables
for metric in metrics:
    # For EDA (equivalent to GSR)
    preferred_eda_mean, preferred_eda_std, non_preferred_eda_mean, non_preferred_eda_std = calculate_group_stats(metric, "GSR")
    
    # For PPG (equivalent to ECG)
    preferred_ppg_mean, preferred_ppg_std, non_preferred_ppg_mean, non_preferred_ppg_std = calculate_group_stats(metric, "ECG")

    # Format the results and add to the EDA table
    eda_table.at[metric["name"], "Preferred EDA (Mean ± SD)"] = f"{preferred_eda_mean:.2f} ± {preferred_eda_std:.2f}"
    eda_table.at[metric["name"], "Non-Preferred EDA (Mean ± SD)"] = f"{non_preferred_eda_mean:.2f} ± {non_preferred_eda_std:.2f}"
    
    # Format the results and add to the PPG table
    ppg_table.at[metric["name"], "Preferred PPG (Mean ± SD)"] = f"{preferred_ppg_mean:.2f} ± {preferred_ppg_std:.2f}"
    ppg_table.at[metric["name"], "Non-Preferred PPG (Mean ± SD)"] = f"{non_preferred_ppg_mean:.2f} ± {non_preferred_ppg_std:.2f}"

# Display the result tables
print(eda_table)
print(ppg_table)

# Export the EDA table to LaTeX format
latex_eda_output = eda_table.to_latex(index=True, column_format='|l|c|c|', 
                                      header=True, bold_rows=True, 
                                      caption='Group-Level Analysis of Preferred and Non-Preferred EDA Conditions',
                                      label='tab:eda_group_analysis')

# Save the EDA LaTeX table to a file
latex_eda_file_path = os.path.join(base_path, "group_analysis_eda_table.tex")
with open(latex_eda_file_path, "w") as f:
    f.write(latex_eda_output)

# Export the PPG table to LaTeX format
latex_ppg_output = ppg_table.to_latex(index=True, column_format='|l|c|c|', 
                                      header=True, bold_rows=True, 
                                      caption='Group-Level Analysis of Preferred and Non-Preferred PPG Conditions',
                                      label='tab:ppg_group_analysis')

# Save the PPG LaTeX table to a file
latex_ppg_file_path = os.path.join(base_path, "group_analysis_ppg_table.tex")
with open(latex_ppg_file_path, "w") as f:
    f.write(latex_ppg_output)

# Print the LaTeX outputs so you can copy them directly from the console
print("EDA LaTeX Table:")
print(latex_eda_output)

print("PPG LaTeX Table:")
print(latex_ppg_output)
