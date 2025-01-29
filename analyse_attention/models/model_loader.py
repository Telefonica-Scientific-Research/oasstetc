import numpy as np
from transformers import (
    AutoTokenizer,
    AutoModelForSequenceClassification,
    AutoModelForCausalLM,
    AutoModelForMaskedLM,
    TFBertForMaskedLM,
    BertTokenizer,
)

from transformers import (
    BatchEncoding,
)

from models.reward_models.llama_rm import LlamaRewardModel
from models.reward_models.qrmllama_rm import LlamaForRewardModelWithGating
from models.reward_models.eurus_rm import EurusRewardModel


class ModelLoaderCausal:
    def load_tokenizer(self, model_name):
        tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
        return tokenizer

    def load_model(self, model_name):
        # Load model directly
        print("Loading causalLM model")
        model = AutoModelForCausalLM.from_pretrained(
            model_name,
            device_map="auto",
            trust_remote_code=True,
            output_attentions=True,
        )
        return model


class ModelLoaderReward:
    def load_tokenizer(self, model_name):
        tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)

        return tokenizer

    def load_model(self, model_name):
        # Load model directly
        print("Loading ModelLoaderReward model")
        model = AutoModelForSequenceClassification.from_pretrained(
            model_name,
            device_map="auto",
            trust_remote_code=True,
            output_attentions=True,
        )
        return model


class ModelLoaderUltra:
    def load_tokenizer(self, model_name):
        tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)

        return tokenizer

    def load_model(self, model_name):
        # Load model directly
        print("Loading ModelLoaderUltra model")
        model = LlamaRewardModel.from_pretrained(
            model_name, trust_remote_code=True, output_attentions=True
        )
        return model


class ModelLoaderQRLlama:
    def load_tokenizer(self, model_name):
        tokenizer = AutoTokenizer.from_pretrained(
            model_name, trust_remote_code=True, use_fast=True
        )

        return tokenizer

    def load_model(self, model_name):
        # Load model directly
        print("Loading ModelLoaderQRLlama model")
        model = LlamaForRewardModelWithGating.from_pretrained(
            model_name,
            device_map="auto",
            trust_remote_code=True,
            output_attentions=True,
        )
        return model


class ModelLoaderEurus:
    def load_tokenizer(self, model_name):
        tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)

        return tokenizer

    def load_model(self, model_name):
        # Load model directly
        print("Loading ModelLoaderEurus model")
        model = EurusRewardModel.from_pretrained(
            model_name,
            trust_remote_code=True,
            output_attentions=True,
        )
        return model


class ModelLoaderBert:
    def load_tokenizer(self, model_name):
        # tokenizer = BertTokenizer.from_pretrained(model_name, use_fast=True)
        tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)

        return tokenizer

    def load_model(self, model_name):
        # Load model directly
        # model = TFBertForMaskedLM.from_pretrained(model_name, output_attentions=True)
        model = AutoModelForMaskedLM.from_pretrained(model_name, output_attentions=True)
        return model


class ModelLoaderFactory:
    def get_model_loader(self, loader_type):
        if loader_type == "BertBased":
            return ModelLoaderBert()
        if loader_type == "causalLM":
            return ModelLoaderCausal()
        elif loader_type == "reward":
            return ModelLoaderReward()
        elif loader_type == "ultraRM":
            return ModelLoaderUltra()
        elif loader_type == "QRLlama":
            return ModelLoaderQRLlama()
        elif loader_type == "eurus":
            return ModelLoaderEurus()

        else:
            raise ValueError(f"Unknown model loader type: {loader_type}")
