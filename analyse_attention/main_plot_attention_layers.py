import os

# Set the timeout to 5 seconds
os.environ["PYDEVD_WARN_SLOW_RESOLVE_TIMEOUT"] = "5.0"

from models.compare_att import CompareAttention

path = "oasstetc_data/"
# models = ['nicolinho/QRM-Llama3.1-8B']
models = {
    "microsoft/phi-1_5": "causalLM",
    "mistralai/Mistral-7B-v0.1": "causalLM",
    "mistralai/Mistral-7B-Instruct-v0.1": "causalLM",
    "meta-llama/Meta-Llama-3-8B": "causalLM",
    "meta-llama/Meta-Llama-3-8B-Instruct": "causalLM",
    "meta-llama/Llama-2-7b-hf": "causalLM",
    "meta-llama/Llama-2-7b-chat-hf": "causalLM",
    "meta-llama/Llama-3.1-8B-Instruct": "causalLM",
    "meta-llama/Llama-3.1-8B": "causalLM",
    "openbmb/UltraRM-13b": "ultraRM",
    "nicolinho/QRM-Llama3.1-8B": "QRLlama",
    "openbmb/Eurus-RM-7b": "eurus",
    "google-bert/bert-base-uncased": "BertBased",
    "google-bert/bert-large-uncased": "BertBased",
    "google-bert/bert-base-cased": "BertBased",
    "FacebookAI/roberta-base": "BertBased",
    "FacebookAI/roberta-large": "BertBased",
}
gaze_features = [
    "fix_duration_n",
    # "fix_duration",
    # "first_fix_duration",
    "first_fix_duration_n",
    "fix_number",
]

gaze_features_names = {
    "fix_duration_n": "TRT",
    "fix_duration": "TRT",
    "first_fix_duration": "FFD",
    "first_fix_duration_n": "FFD",
    "fix_number": "nFix",
}
# filter_completed = True
filter_completed = True
path = "oasstetc_data/"
folder_attention = "attention"
# ----------------------------------------

for model_name, model_type in models.items():
    print(model_name)
    CompareAttention(
        model_name=model_name, model_type=model_type, path=path
    ).plot_attention_all_trials(
        gaze_features=gaze_features, attention_folder="attention", filter_completed=True
    )
