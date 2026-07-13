import os

# Directory holding locally downloaded model weights; set to "" to always pull from the hub.
LOCAL_MODEL_DIR = os.environ.get("LOCAL_MODEL_DIR", "/data/USER/lbw2026/llm/model")


def resolve_model_path(model_id):
    """
    Prefers a locally downloaded copy of the model under LOCAL_MODEL_DIR
    (matched by repo basename, e.g. meta-llama/Meta-Llama-3-8B-Instruct ->
    LOCAL_MODEL_DIR/Meta-Llama-3-8B-Instruct); falls back to the hub id.
    """
    if LOCAL_MODEL_DIR:
        local_path = os.path.join(LOCAL_MODEL_DIR, model_id.split("/")[-1])
        if os.path.isfile(os.path.join(local_path, "config.json")):
            print(f"Loading model from local path: {local_path}")
            return local_path
    print(f"Local copy not found, loading from hub: {model_id}")
    return model_id


def load_tokenizer(model_path, tokenizer_cls=None):
    """
    Loads a tokenizer, falling back to the slow implementation when the
    installed tokenizers version cannot parse newer tokenizer.json files.
    """
    from transformers import AutoTokenizer
    tokenizer_cls = tokenizer_cls or AutoTokenizer
    try:
        return tokenizer_cls.from_pretrained(model_path)
    except Exception:
        return tokenizer_cls.from_pretrained(model_path, use_fast=False)
