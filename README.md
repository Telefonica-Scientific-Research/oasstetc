# OASSTETC


## Dependencies
- See requirements.txt for full list of dependencies
- Key packages include:
  - OpenCV
  - NumPy
  - Pandas
  - HuggingFace Hub
  - PyTorch


### process_et_data and generate reading measures
    - Main functionality for processing eye tracking data and generating reading measures
    - Contains classes for handling text and image-based eye tracking data

    ## Assign fixations to words:

    For each user and session, we have:
    * 1 file with fixation data: one row for each fixation. We read X, Y coordinates to position it on the screen. We use features such as the number of seconds and pupil dilation.

    * N images, one for each trial (response to a prompt). Each image has a name indicating the user, session, and trial. Each image displays the user's response. We use these images to obtain the X, Y coordinates for each word.

    * N CSV files, one for each trial. It contains the X, Y coordinates for each trial. Initially used to obtain word coordinates, but later it was decided to use images.

    * 1 file with trial data: one row for each trial. We obtain the original text and user ratings.

    The class EyeTrackingDataText(EyeTrackingData) (in the file eye_tracking_data.py) is used to obtain word coordinates from CSV with X, Y coordinates for each character. It was an initial approach, but later it was decided to use images.

    The class EyeTrackingDataImage(EyeTrackingData) (in the file eye_tracking_data_image.py) reads fixation and image data. From images with OCR, we obtain word coordinates. The class also includes an algorithm to assign fixations to words. Once we have word coordinates (two on the X-axis and two on the Y-axis), we assign fixations, considering the minimum distance of each fixation to each word and the assignment of the previous fixation. The output is an CSV file with one row per word (or set of words, depending on how OCR extracted them) and one column per feature (number of fixations, sum of fixations in seconds, mean of fixations in seconds, pupil dilation).

    * It allows to extract words from images with OCR, read fixations files, assign fixations to words, and save the results in a CSV file.
    * It allows plotting images with prompts and fixations assigned to each word.
    
    run `python process_et_data/main_asign_fixations.py` to process the raw eye tracking data,  assign fixations to words and compute reading measures per word and per user
    run `python process_et_data/main.py` to compute reading measures per user and per user set
        (EyeTrackingAnalyser().average_gaze_features_real_participants(path, path_save))

### generate_syntethic_readingmeasures
    Code to generate synthetic fixations for the text used in the lab and then be able to compare it. 
    run `python generate_syntethic_readingmeasures/main_generate_synthetic_fixations.py` to generate the synthetic fixations

### analyse_reading_measures
    Code to analyse the reading measures obtained from the eye tracking data.

    This module analyzes reading measures extracted from eye-tracking data. It includes processing real and synthetic eye-tracking metrics, computing statistical comparisons, and visualizing results.
    
    Features of the scripts:
    - Processes eye-tracking data to compute reading measures, such as fixation duration, first fixation duration, and fixation count.
    - Compares reading metrics between preferred and non-preferred conditions.
    - Computes statistical tests (paired) to assess differences between conditions.
    - Supports real and synthetic data analysis.
    - Synchronizes physiological data (EDA) with eye-tracking data to analyze correlations.
    - Produce boxplot figures as seen in Figure 3/4/5. 

    run `python analyse_readingmeasures/Figure_3_Total_words_and_length_by_condition.py` to generate Figure 3 (Total words and length by condition)
    run `python analyse_readingmeasures/Figure_4_and_5_Reading_measures_Sythetic_and_organic_data.py` to generate Figure 4/5 (Reading measurements in synthetic and organic data)
    run `analyse_readingmeasures/Table_7_EDA_analysis.py` to generate table 7 (EDA analysis)
   

### analyse_attention
    ## Compute attention of each model:
    - Attention analysis for language models
    - Supports multiple models:
        run `python analyse_attention/main_compute_attention.py` 
    - With this models you  can compute the attention of the response of the combination fo the promtp and the response (for the reward model):
        * openbmb/UltraRM-13b (ultraRM)
        * openbmb/Eurus-RM-7b (eurus)
        * nicolinho/QRM-Llama3.1-8B (QRLlama)



