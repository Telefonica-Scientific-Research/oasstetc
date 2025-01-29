import os

# Set the timeout to 5 seconds
os.environ["PYDEVD_WARN_SLOW_RESOLVE_TIMEOUT"] = "5.0"
import argparse
from models.compare_att import CompareAttention

# CUDA_VISIBLE_DEVICE=2 python main_compare_trials.py  --folder_attention attention --filter_completed False
# CUDA_VISIBLE_DEVICE=1 python main_compare_trials.py  --folder_attention attention_reward --filter_completed False
# CUDA_VISIBLE_DEVICE=3 python main_compare_trials.py  --folder_attention attention_reward --filter_completed True
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
    args = parser.parse_args()
    print(f"Filter completed variable :{str(args.filter_completed).lower()}")
    filter_completed = str(args.filter_completed).lower() == "true"
    print(f"Filter variable:{filter_completed, type(filter_completed)}")
    folder_attention = args.folder_attention

    # folder_attention = "attention"
    # filter_completed = False

    path = "oasstetc_data/"
    models = {
        # -----------------------------------------------
        # "google-bert/bert-base-uncased": "BertBased",
        # "google-bert/bert-large-uncased": "BertBased",
        # "google-bert/bert-base-cased": "BertBased",
        # "FacebookAI/roberta-base": "BertBased",
        # "FacebookAI/roberta-large": "BertBased",
        # # -----------------------------------------------
        # "mistralai/Mistral-7B-v0.1": "causalLM",
        # "mistralai/Mistral-7B-Instruct-v0.1": "causalLM",
        # "meta-llama/Llama-2-7b-hf": "causalLM",
        # "meta-llama/Llama-2-7b-chat-hf": "causalLM",
        # # -----------------------------------------------
        # "meta-llama/Meta-Llama-3-8B": "causalLM",
        # "meta-llama/Meta-Llama-3-8B-Instruct": "causalLM",
        # "meta-llama/Llama-3.1-8B": "causalLM",
        # "meta-llama/Llama-3.1-8B-Instruct": "causalLM",
        # # -----------------------------------------------
        # "microsoft/phi-1_5": "causalLM",
        "openbmb/UltraRM-13b": "ultraRM",
        "openbmb/Eurus-RM-7b": "eurus",
        "nicolinho/QRM-Llama3.1-8B": "QRLlama",
        # #-----------------------------------------------
    }
    gaze_features = [
        "fix_duration_n",
        # "fix_duration",
        # "first_fix_duration",
        "first_fix_duration_n",
        "fix_number",
    ]
    # compute per all
    # ----------------------------------------
    # for gaze_feature in gaze_features:
    #     for model_name, model_type in models.items():
    #         print(model_name)
    #         sc_users = CompareAttention(
    #             model_name=model_name, model_type=model_type, path=path
    #         ).compute_sc_model_per_trial(
    #             gaze_feature=gaze_feature, filter_completed=filter_completed, folder_attention=folder_attention,
    #         )
    #     # compute results of all models
    #     CompareAttention.compare_between_models_per_trials(
    #         folder=path + folder_attention + "/",
    #         gaze_feature=gaze_feature,
    #         filter_completed=filter_completed,
    #     )
    # ----------------------------------------

    # compute per chosen an rejected
    # ----------------------------------------
    for gaze_feature in gaze_features:
        # print(f"Compute per chosen and rejected for gaze feature:{gaze_feature}")
        for model_name, model_type in models.items():
            print(model_name)
            sc_users = CompareAttention(
                model_name=model_name, model_type=model_type, path=path
            ).compute_sc_model_chosenrejected_per_trials(
                gaze_feature=gaze_feature,
                filter_completed=filter_completed,
                folder_attention=folder_attention,
            )
        # # compute results of all models
        CompareAttention.compare_between_models_chosenrejected_per_trials(
            folder=path + folder_attention + "/",
            gaze_feature=gaze_feature,
            filter_completed=filter_completed,
        )
    # ----------------------------------------
