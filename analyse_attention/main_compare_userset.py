import os

# Set the timeout to 5 seconds
os.environ["PYDEVD_WARN_SLOW_RESOLVE_TIMEOUT"] = "5.0"
import argparse
from models.compare_att import CompareAttention

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="")
    parser.add_argument(
        "-f",
        "--filter_completed",
        default=False,
    )
    parser.add_argument(
        "--folder_attention",
        type=str,
        default="attention",
    )
    path = "oasstetc_data/"
    args = parser.parse_args()
    filter_completed = str(args.filter_completed).lower() == "true"
    folder_attention = args.folder_attention

    models = {
        # -----------------------------------------------
        "google-bert/bert-base-uncased": "BertBased",
        "google-bert/bert-large-uncased": "BertBased",
        "google-bert/bert-base-cased": "BertBased",
        "FacebookAI/roberta-base": "BertBased",
        "FacebookAI/roberta-large": "BertBased",
        # -----------------------------------------------
        "mistralai/Mistral-7B-v0.1": "causalLM",
        "mistralai/Mistral-7B-Instruct-v0.1": "causalLM",
        "meta-llama/Llama-2-7b-hf": "causalLM",
        "meta-llama/Llama-2-7b-chat-hf": "causalLM",
        # # -----------------------------------------------
        # "meta-llama/Meta-Llama-3-8B": "causalLM",
        # "meta-llama/Meta-Llama-3-8B-Instruct": "causalLM",
        # "meta-llama/Llama-3.1-8B": "causalLM",
        # "meta-llama/Llama-3.1-8B-Instruct": "causalLM",
        # # -----------------------------------------------
        # "microsoft/phi-1_5": "causalLM",
        # "openbmb/UltraRM-13b": "ultraRM",
        # "openbmb/Eurus-RM-7b": "eurus",
        # "nicolinho/QRM-Llama3.1-8B": "QRLlama",
        # #-----------------------------------------------
    }
    gaze_features = [
        "fix_duration_n",
        "fix_duration",
        "first_fix_duration",
        "first_fix_duration_n",
        "fix_number",
    ]

    # compute per all
    # ----------------------------------------
    for gaze_feature in gaze_features:
        for model_name, model_type in models.items():
            print(model_name)
            sc_users = CompareAttention(
                model_name=model_name, model_type=model_type, path=path
            ).compute_sc_model_per_userset(
                gaze_feature=gaze_feature,
                filter_completed=filter_completed,
                folder_attention=folder_attention,
            )
        # compute results of all models
        CompareAttention.compare_between_models_per_userset(
            folder=path + folder_attention + "/",
            gaze_feature=gaze_feature,
            filter_completed=filter_completed,
        )
    # ----------------------------------------

    # compute per chosen an rejected
    # ----------------------------------------
    for gaze_feature in gaze_features:
        for model_name, model_type in models.items():
            print(model_name)
            sc_users = CompareAttention(
                model_name=model_name, model_type=model_type, path=path
            ).compute_sc_model_chosenrejected_per_userset(
                gaze_feature=gaze_feature,
                filter_completed=filter_completed,
                folder_attention=folder_attention,
            )
        # compute results of all models
        CompareAttention.compare_between_models_chosenrejected_per_userset(
            folder=path + folder_attention + "/",
            gaze_feature=gaze_feature,
            filter_completed=filter_completed,
        )
    # ----------------------------------------
