import os
import sys
import pathlib
import pandas as pd

sys.path.append("../..")
path = str(pathlib.Path(__file__).parent.resolve())
sys.path.append(path)
path = str(pathlib.Path(__file__).parent.resolve().parent.resolve())
sys.path.append(path)
path = str(pathlib.Path(__file__).parent.resolve().parent.resolve().parent.resolve())
sys.path.append(path)
cwd = os.getcwd()
import numpy as np

sys.path.append(cwd)

from eyetrackpy.data_generator.fixations_predictor.models.generate_fixations_predictor import (
    CreateFixationsPredictorModel,
)
from eyetrackpy.data_generator.models.fixations_aligner import FixationsAligner

from eyetrackpy.data_processor.models.eye_tracking_data_image import (
    EyeTrackingDataImage,
)
from eyetrackpy.data_processor.models.eye_tracking_data_simple import (
    EyeTrackingDataUserSet,
) 


def save_word_fixations_syn_trial(folder_name, trial, words_fix_trial):
    """
    Save coordinates of words of trial in csv file"""
    # create_csv with words_fix_trial
    words_fix_trial.to_csv(
        folder_name + "/" + "/word_gaze_synthetic_" + str(trial) + ".csv",
        sep=";",
    )
    return True


if __name__ == "__main__":
    
    create_fixations = CreateFixationsPredictorModel(version=2, remap=False)
    # load lab data and compute sentences
    # be carefull with the text and the words generate with the OR
    parent_path = str(pathlib.Path(__file__).parent.resolve().parent.resolve())
    path = parent_path + "/oasstetc/gaze_features_real/"
    path_syn = parent_path + "/oasstetc/gaze_features_synthetic/"

    datauserset = EyeTrackingDataUserSet()

    for user_set in range(1, 9):
        folder = path + "set_" + str(user_set) + "/"
        folder_syn = path_syn + "set_" + str(user_set) + "/"
        files = datauserset.search_word_coor_fixations_files(folder)
        for trial, file in files.items():
            coordinates_data = datauserset._read_coor_trial(file)
            list_words_first = list(coordinates_data.text)
            list_words_first = [str(x) for x in list_words_first]
            text = " ".join(list_words_first)

            (
                fixations,
                fixations_attention_mask,
                _,
                _,
                text_tokenized_fix,
                sentences,
            ) = create_fixations.FP_model._compute_mapped_fixations(sentences=text)
            # align to the words of the original text
            features = np.transpose(fixations.detach().cpu().numpy()).squeeze()
            list_words_first = [x.strip().lower() for x in list_words_first]
            features_mapped = FixationsAligner().map_features_from_words_to_words(
                list_words_first=list_words_first,
                text=text,
                text_tokenized=text_tokenized_fix,
                features=features,
                mode="max",
            )
            name_features = [
                "fix_number",
                "first_fix_duration",
                "GPT",
                "fix_duration",
                "fixProp",
            ]
            features_mapped = dict(zip(name_features, features_mapped))
            features_mapped["text"] = list_words_first
            features_mapped = pd.DataFrame(features_mapped)
            # check if folder_syn exists
            if not os.path.exists(folder_syn):
                os.makedirs(folder_syn)
            save_word_fixations_syn_trial(folder_syn, trial, features_mapped)
