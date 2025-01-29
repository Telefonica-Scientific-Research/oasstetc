import json
import pandas as pd
from eyetrackpy.data_processor.models.eye_tracking_data_simple import (
    EyeTrackingDataUserSet,
)
import os


class HumanAttentionExtractor:
    def load_gaze_features(self, folder):
        datauserset = EyeTrackingDataUserSet()
        files = datauserset.search_word_coor_fixations_files(folder)
        fixations_trials = {}
        for trial, file in files.items():
            fixations_trials[trial] = datauserset._read_coor_trial(file)
        return fixations_trials

    def load_gaze_features_all(self, path="oasstetc_data/gaze_features_real/"):
        fixations_trials_all = {}
        for user_set in range(1, 9):
            folder = path + "set_" + str(user_set) + "/"
            fixations_trials_all[user_set] = self.load_gaze_features(user_set, folder)
        return fixations_trials_all

    def load_texts(self, path):
        fixations_trials = self.load_gaze_features(path)
        return {
            trial: list(fixations_trial.text)
            for trial, fixations_trial in fixations_trials.items()
        }

    @staticmethod
    def load_trial_prompts(path="oasstetc_data/raw_data/"):
        texts_prompts = pd.read_csv(path + "text_prompts.csv", sep=";")
        return texts_prompts

    @staticmethod
    def load_trials_info(path):
        with open(path + "info_trials.json", "r") as file:
            return json.load(file)
