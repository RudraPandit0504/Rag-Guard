import time
from huggingface_hub import hf_hub_download

REPO = "sentence-transformers/all-MiniLM-L6-v2"

FILES = [
    "config.json",
    "config_sentence_transformers.json",
    "modules.json",
    "sentence_bert_config.json",
    "tokenizer.json",
    "tokenizer_config.json",
    "vocab.txt",
    "special_tokens_map.json",
    "model.safetensors",
    "1_Pooling/config.json",
]

for name in FILES:
    for attempt in range(1, 9):
        try:
            path = hf_hub_download(repo_id=REPO, filename=name)
            print(f"OK   {name}")
            break
        except Exception as e:
            print(f"  retry {attempt} on {name}: {type(e).__name__}")
            time.sleep(2)
    else:
        print(f"FAIL {name}")

print("\ndone")