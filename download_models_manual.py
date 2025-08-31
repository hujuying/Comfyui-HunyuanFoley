`ComfyUI/models/foley` directory. Files are fetched from the official
HuggingFace repository using simple HTTP requests.
"""
import os
import pathlib
import requests
from typing import Dict, List

# Mapping of model filenames to their download URLs
MODEL_URLS: Dict[str, str] = {
    "hunyuanvideo_foley.pth": "https://huggingface.co/tencent/HunyuanVideo-Foley/resolve/main/hunyuanvideo_foley.pth",
    "synchformer_state_dict.pth": "https://huggingface.co/tencent/HunyuanVideo-Foley/resolve/main/synchformer_state_dict.pth",
    "vae_128d_48k.pth": "https://huggingface.co/tencent/HunyuanVideo-Foley/resolve/main/vae_128d_48k.pth",
}


def download_file(url: str, dest: pathlib.Path) -> None:
    """Download a URL to a local path with streaming."""
    dest.parent.mkdir(parents=True, exist_ok=True)
    with requests.get(url, stream=True) as r:
        r.raise_for_status()
        with open(dest, "wb") as f:
            for chunk in r.iter_content(chunk_size=8192):
                if chunk:
                    f.write(chunk)


def download_all(model_dir: str) -> List[pathlib.Path]:
    model_paths = []
    for name, url in MODEL_URLS.items():
        target = pathlib.Path(model_dir) / name
        if target.exists():
            model_paths.append(target)
            continue
        download_file(url, target)
        model_paths.append(target)
    return model_paths


if __name__ == "__main__":
    # Determine default model directory relative to ComfyUI
    from folder_paths import models_dir
    dest_dir = os.path.join(models_dir, "foley")
    paths = download_all(dest_dir)
    for p in paths:
        print(f"Downloaded: {p}")