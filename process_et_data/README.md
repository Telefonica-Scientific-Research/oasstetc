# analyse_eye_tracking_results
## other_analyse

    Notebooks with some initial analyses of the eye-tracking data to figure it out how to use asign fixations to words.
## data
    For each user and session, we have:
        * 1 file with fixation data: one row for each fixation. We read X, Y coordinates to position it on the screen. We use features such as the number of seconds and pupil dilation.

        * N images, one for each trial (response to a prompt). Each image has a name indicating the user, session, and trial. Each image displays the user's response. We use these images to obtain the X, Y coordinates for each word.

        * N CSV files, one for each trial. It contains the X, Y coordinates for each trial. Initially used to obtain word coordinates, but later it was decided to use images.

        * 1 file with trial data: one row for each trial. We obtain the original text and user ratings.

## models
    ### The class EyeTrackingDataText(EyeTrackingData) (in the file eye_tracking_data.py) is used to obtain word coordinates from CSV with X, Y coordinates for each character. It was an initial approach, but later it was decided to use images.

    ### The class EyeTrackingDataImage(EyeTrackingData) (in the file eye_tracking_data_image.py) reads fixation and image data. From images with OCR, we obtain word coordinates. The class also includes an algorithm to assign fixations to words. Once we have word coordinates (two on the X-axis and two on the Y-axis), we assign fixations, considering the minimum distance of each fixation to each word and the assignment of the previous fixation. The output is an CSV file with one row per word (or set of words, depending on how OCR extracted them) and one column per feature (number of fixations, sum of fixations in seconds, mean of fixations in seconds, pupil dilation).

    * It allows to extract words from images with OCR, read fixations files, assign fixations to words, and save the results in a CSV file.
    * It allows plotting images with prompts and fixations assigned to each word.