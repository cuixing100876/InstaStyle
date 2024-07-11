import time
from huggingface_hub import snapshot_download
# huggingface repo id
repo_id = "CompVis/stable-diffusion-v1-4"
# save path
local_dir = "./stable-diffusion-v1-4"

while True:
    try:
        snapshot_download(cache_dir=local_dir,
        local_dir=local_dir,
        repo_id=repo_id,
        local_dir_use_symlinks=False,
        resume_download=True,
        allow_patterns=["*.model", "*.json", "*.bin",
        "*.py", "*.md", "*.txt"],
        ignore_patterns=["*.safetensors", "*.msgpack",
        "*.h5", "*.ot",],
        )
    except Exception as e :
        print(e)
        # time.sleep(5)
    else:
        print('success!')
        break