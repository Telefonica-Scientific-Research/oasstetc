import numpy as np
import scipy.special

import pandas as pd
import numpy as np

import os
import torch

from eyetrackpy.data_generator.models.fixations_aligner import FixationsAligner
from tokenizeraligner.models.tokenizer_aligner import TokenizerAligner
from transformers import (
    BatchEncoding,
)

from models.model_loader import ModelLoaderFactory


class ModelAttentionExtractor:
    def __init__(self, model_name, model_type):
        self.model_type = model_type
        model_loader = ModelLoaderFactory().get_model_loader(model_type)
        self.model = model_loader.load_model(model_name)
        self.tokenizer = model_loader.load_tokenizer(model_name)
        print("tokenizer chat template")
        print(self.tokenizer.chat_template)
        vocab = self.tokenizer.get_vocab()
        self.special_tokens_id = [
            vocab[token] for _, token in self.tokenizer.special_tokens_map.items()
        ]

    @staticmethod
    def map_attention_from_tokens_to_words_v2(
        list_words_first: list,
        text: str,
        text_tokenized: BatchEncoding,
        features_mapped_second_words: list,
        mode="max",
    ):
        # funcon done for when
        # we compute the words list from the text_tokenized
        list_words_second = TokenizerAligner().tokens_to_words(text, text_tokenized)
        # we map the original words (list_word_first) to the words in the text_tokenized
        mapped_words_idxs, mapped_words_str = TokenizerAligner().map_words(
            list_words_first, list_words_second
        )
        # we asign cero to the positions of the special tokens

        features_mapped_first_words = (
            TokenizerAligner().map_features_between_paired_list(
                features_mapped_second_words,
                mapped_words_idxs,
                list_words_first,
                mode=mode,
            )
        )
        return features_mapped_first_words

    @staticmethod
    def map_attention_from_tokens_to_words_reward(
        list_words_first: list,
        text: str,
        text_tokenized: BatchEncoding,
        features_mapped_second_words: list,
        index_init: int = 0,
        mode: str = "mean",
    ):
        # we compute the words list from the text_tokenized
        list_words_second = TokenizerAligner().tokens_to_words(text, text_tokenized)
        # we filtere only the response
        list_words_second = list_words_second[index_init + 1 :]
        features_mapped_second_words = features_mapped_second_words[index_init + 1 :]
        # we map the original words (list_word_first) to the words in the text_tokenized
        mapped_words_idxs, mapped_words_str = TokenizerAligner().map_words(
            list_words_first, list_words_second
        )
        features_mapped_first_words = (
            TokenizerAligner().map_features_between_paired_list(
                features_mapped_second_words,
                mapped_words_idxs,
                list_words_first,
                mode=mode,
            )
        )
        return features_mapped_first_words

    @staticmethod
    def map_attention_from_words_to_words(
        list_words_first: list,
        text: str,
        text_tokenized: BatchEncoding,
        features_mapped_second_words: list,
        mode="max",
    ):
        # we compute the words list from the text_tokenized
        list_words_second = TokenizerAligner().text_to_words(
            text, text_tokenized, text_tokenized.word_ids()
        )
        # we map the original words (list_word_first) to the words in the text_tokenized
        mapped_words_idxs, mapped_words_str = TokenizerAligner().map_words(
            list_words_first, list_words_second
        )
        features_mapped_first_words = (
            TokenizerAligner().map_features_between_paired_list(
                features_mapped_second_words,
                mapped_words_idxs,
                list_words_first,
                mode="mean",
            )
        )
        return features_mapped_first_words

    @staticmethod
    def get_attention_model(model, inputs):
        # check if model has atribute device
        if not hasattr(model, "device"):
            input_ids = inputs["input_ids"]
            attention_mask = inputs["attention_mask"]
        else:
            input_ids = inputs["input_ids"].to(model.device)
            attention_mask = inputs["attention_mask"].to(model.device)
        output = model(input_ids=input_ids, attention_mask=attention_mask)
        return output.attentions

    # For the attention baseline, we fixed several experimental choices (see below) which might affect the results.

    def process_attention_reward(
        self,
        attention,
        input_ids: BatchEncoding,
        text: str = None,
        list_word_original: list = None,
        index_init: int = 0,
        method="sum",
    ):
        attention_layer = {}
        special_token_idx = self.compute_special_token_idx(
            input_ids["input_ids"][0].tolist(), self.special_tokens_id
        )
        for layer in range(len(attention)):
            att_layer = attention[layer][0].cpu().detach().numpy()
            mean_attention = np.mean(att_layer, axis=0)
            if method == "sum":
                aggregated_attention = np.sum(mean_attention, axis=0)
            else:
                if self.model_type == "BertBased":
                    aggregated_attention = np.mean(mean_attention, axis=0)
                else:
                    # we normalize the attention taking into account the decoder nature and the masking
                    # mean_attention_scaled_decoder = self.normalize_rows(mean_attention)
                    # we dont want to include the ceros because of masking
                    aggregated_attention = self.compute_mean_diagonalbewlow(
                        mean_attention
                    )
            aggregated_attention = [
                0 if i in special_token_idx else aggregated_attention[i]
                for i in range(len(aggregated_attention))
            ]
            aggregated_attention = self.map_attention_from_tokens_to_words_reward(
                list_word_original,
                text,
                input_ids,
                features_mapped_second_words=aggregated_attention,
                index_init=index_init,
                mode="mean",
            )
            # we ave without this and do it later
            relative_attention = scipy.special.softmax(aggregated_attention)
            # relative_attention = aggregated_attention
            attention_layer[layer] = relative_attention
        return attention_layer

    def process_attention(
        self,
        attention,
        input_ids: BatchEncoding,
        text: str = None,
        list_word_original: list = None,
        word_level: bool = True,
        method="sum",
    ):
        attention_layer = {}
        special_token_idx = self.compute_special_token_idx(
            input_ids["input_ids"][0].tolist(), self.special_tokens_id
        )
        for layer in range(len(attention)):
            att_layer = attention[layer][0].cpu().detach().numpy()
            mean_attention = np.mean(att_layer, axis=0)
            # For each token, we sum over the attentions received from all other tokens.
            if method == "sum":
                aggregated_attention = np.sum(mean_attention, axis=0)
            else:
                if self.model_type == "BertBased":
                    aggregated_attention = np.mean(mean_attention, axis=0)
                else:
                    # we normalize the attention taking into account the decoder nature and the masking
                    # mean_attention_scaled_decoder = self.normalize_rows(mean_attention)
                    # we dont want to include the ceros because of masking
                    aggregated_attention = self.compute_mean_diagonalbewlow(
                        mean_attention
                    )

            if word_level is True:
                if list_word_original is None:
                    raise ValueError(
                        "list_word_original must be provided when word_level is True"
                    )
                if text is None:
                    raise ValueError("text must be provided when word_level is True")
                aggregated_attention = [
                    0 if i in special_token_idx else aggregated_attention[i]
                    for i in range(len(aggregated_attention))
                ]
                aggregated_attention_mapped_words = (
                    FixationsAligner().map_features_from_tokens_to_words(
                        aggregated_attention, input_ids, mode="sum"
                    )
                )
                if aggregated_attention_mapped_words is None:
                    aggregated_attention = self.map_attention_from_tokens_to_words_v2(
                        list_word_original,
                        text,
                        input_ids,
                        features_mapped_second_words=aggregated_attention,
                        mode="mean",
                    )
                else:
                    aggregated_attention = self.map_attention_from_words_to_words(
                        list_word_original,
                        text,
                        input_ids,
                        aggregated_attention_mapped_words,
                        mode="mean",
                    )

            else:
                # to compute attention per token we remove special tokens
                aggregated_attention = np.delete(
                    aggregated_attention, special_token_idx
                )
            relative_attention = scipy.special.softmax(aggregated_attention)
            # relative_attention = aggregated_attention
            attention_layer[layer] = relative_attention
        return attention_layer

    def extract_attention(self, texts_trials: dict, word_level: bool = True):
        attention_trials = {}
        for trial, list_text in texts_trials.items():
            print("trial", trial)
            list_word_original = [str(x) for x in list_text]
            text = " ".join(list_word_original)
            list_word_original = [x.lower() for x in list_word_original]
            input_ids = self.tokenize_text(self.tokenizer, text)
            attention = self.get_attention_model(self.model, input_ids)
            try:
                attention_trials[trial] = self.process_attention(
                    attention,
                    input_ids,
                    text=text,
                    list_word_original=list_word_original,
                    word_level=word_level,
                )
            except Exception as e:
                print(trial, "error:", e)
        return attention_trials

    @staticmethod
    def tokenize_text_chat(tokenizer, prompt, response, model_type="ULTRA"):
        if model_type == "ultraRM":
            text = "Human: " + prompt + "\nAssistant: " + response
        elif model_type == "QRLlama":
            messages = [
                {"role": "user", "content": prompt},
                {"role": "assistant", "content": response},
            ]
            tokens = tokenizer.apply_chat_template(
                messages, return_tensors="pt", return_offsets_mapping=True
            )
            text = tokenizer.decode(tokens[0])
        elif model_type == "eurus":
            text = "[INST] " + prompt + " [/INST] " + response
        else:
            return None

        return text

    @staticmethod
    def find_last_consecutive_pair(nums, tokenizer, model_type):
        # def find_last_consecutive_pair_qrllama(nums, num0=128006, num1=78191, num2=128007):
        #     last_index = -1  # Initialize to -1 to indicate no pair found yet
        #     for i in range(len(nums) - 1):
        #         if nums[i] == num0 and nums[i + 1] == num1 and nums[i + 2] == num2:
        #             last_index = i + 2  # Update last_index to the current index
        #     return last_index

        def find_sequence_last_index(lst, sequence):
            n = len(sequence)

            for i in range(len(lst) - n + 1):
                if lst[i : i + n] == sequence:
                    return i + n - 1

            return -1  # Return -1 if the sequence is not found

        def find_last_consecutive_pair_token(
            ids, tokenizer, text_tokens=" [/INST] ", tokens_id=[]
        ):
            if tokens_id == []:
                ids_chat_end = tokenizer(text_tokens, add_special_tokens=False)
                tokens_id = ids_chat_end["input_ids"]

            last_index = find_sequence_last_index(ids.tolist(), tokens_id)
            return last_index

        if model_type == "ultraRM":
            last_index = find_last_consecutive_pair_token(
                nums,
                tokenizer,
                text_tokens="\nAssistant:",
                # tokens_id=[4007, 22137, 29901],
                tokens_id=[7900, 22137, 29901],
            )
        elif model_type == "QRLlama":
            last_index = find_last_consecutive_pair_token(
                nums, tokenizer, text_tokens="assistant<|end_header_id|>\n\n"
            )
        elif model_type == "eurus":
            last_index = find_last_consecutive_pair_token(
                nums, tokenizer, text_tokens="[/INST]"
            )
        else:
            return -1
        return last_index

    @staticmethod
    def normalize_rows(matrix):
        cols_normalized = []
        for row in range(matrix.shape[0]):
            elements = matrix[row,]
            cols_normalized.append(elements / (1 / (row + 1)))
        return np.array(cols_normalized)

    @staticmethod
    def compute_mean_diagonalbewlow(matrix):
        means = []
        for col in range(matrix.shape[1]):
            # Select elements in the current column from the diagonal and below
            valid_elements = matrix[col:, col]  # Start from the current row downwards
            means.append(
                np.mean(valid_elements)
            )  # Calculate the mean of these elements
        return np.array(means)

    def extract_attention_reward(self, texts_trials: dict, texts_promps: pd.DataFrame):
        attention_trials = {}
        for trial, list_text in texts_trials.items():
            print("trial", trial)
            list_word_original = [str(x) for x in list_text]
            text = " ".join(list_word_original)
            list_word_original = [x.lower() for x in list_word_original]
            prompt = texts_promps[texts_promps["n_resp"] == float(trial)][
                "prompt_text"
            ].values[0]
            text_chat = self.tokenize_text_chat(
                self.tokenizer, prompt, text, model_type=self.model_type
            )
            input_ids = self.tokenize_text(self.tokenizer, text_chat)
            index_init = self.find_last_consecutive_pair(
                input_ids["input_ids"][0],
                tokenizer=self.tokenizer,
                model_type=self.model_type,
            )
            attention = self.get_attention_model(self.model, input_ids)
            # try:
            attention_trials[trial] = self.process_attention_reward(
                attention,
                input_ids,
                text=text_chat,
                list_word_original=list_word_original,
                index_init=index_init,
            )
            # except Exception as e:
            #     print(trial, "error:", e)
        return attention_trials

    @staticmethod
    def compute_special_token_idx(tokens_list, special_tokens_ids):
        special_token_idx = []
        for i, token in enumerate(tokens_list):
            if token in special_tokens_ids:
                special_token_idx.append(i)
        return special_token_idx

    @staticmethod
    def tokenize_text(tokenizer, text):
        return tokenizer(
            text,
            return_tensors="pt",
            add_special_tokens=True,
            padding=False,
            truncation=False,
            return_offsets_mapping=True,
        )

    @staticmethod
    def save_attention_np(attention_trials, path_folder):
        for trial, attention_layer in attention_trials.items():
            path_folder_trial = path_folder + "trial_" + str(trial)
            if not os.path.exists(path_folder_trial):
                os.makedirs(path_folder_trial)
            for layer, attention in attention_layer.items():
                np.save(path_folder_trial + "/layer_" + str(layer) + ".npy", attention)

    @staticmethod
    def save_attention_df(attention_trials, texts_trials, path_folder):
        for trial, attention_layer in attention_trials.items():
            path_folder_trial = path_folder + "trial_" + str(trial)
            if not os.path.exists(path_folder_trial):
                os.makedirs(path_folder_trial)
            trial_text = texts_trials[trial]
            for layer, attention in attention_layer.items():
                pd.DataFrame({"text": trial_text, "attention": attention}).to_csv(
                    path_folder_trial + "/layer_" + str(layer) + ".csv",
                    sep=";",
                    index=False,
                )

    @staticmethod
    def load_attention_np(path_folder):
        attention_trials = {}
        for trial in os.listdir(path_folder):
            attention_layer = {}
            for layer in os.listdir(path_folder + "/" + trial):
                attention = np.load(path_folder + "/" + trial + "/" + layer)
                attention_layer[int(layer.split("_")[1].split(".")[0])] = attention
            attention_trials[float(trial.split("_")[1])] = attention_layer
        return attention_trials

    @staticmethod
    def load_attention_df(path_folder):
        attention_trials = {}
        for trial in os.listdir(path_folder):
            attention_layer = {}
            for layer in os.listdir(path_folder + "/" + trial):
                attention = pd.read_csv(
                    path_folder + "/" + trial + "/" + layer, sep=";"
                )
                attention_layer[int(layer.split("_")[1].split(".")[0])] = attention
            attention_trials[float(trial.split("_")[1])] = attention_layer
        return attention_trials
