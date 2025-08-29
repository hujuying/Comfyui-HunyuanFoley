# ComfyUI_HunyuanFoley/nodes.py
# MMAudio-style nodes for Tencent HunyuanVideo-Foley

import os, sys, tempfile
from typing import Optional

import torch
import numpy as np
import imageio

import folder_paths
import comfy.model_management as mm

# If you bundled the repo code in this custom_nodes folder:
sys.path.insert(0, os.path.dirname(__file__))

# Hunyuan internals
from hunyuanvideo_foley.utils.model_utils import load_model, denoise_process
from hunyuanvideo_foley.utils.feature_utils import feature_process

# --------------------------------------------------------------------------------
# Register BOTH names so your picker works no matter the folder casing you used
#   models/hunyuanFoley  and  models/hunyuanfoley
# Use "hunyuanFoley" as the primary key for UI dropdowns.
# --------------------------------------------------------------------------------
_models_root = folder_paths.models_dir
_primary_key = "hunyuanFoley"
_alt_key     = "hunyuanfoley"

def _ensure_model_key(key_name: str, subdir: str):
    if key_name not in folder_paths.folder_names_and_paths:
        folder_paths.add_model_folder_path(key_name, os.path.join(_models_root, subdir))

# Try to map both keys to wherever you actually put the files
# If only one of the folders exists, point both keys to that one.
path_A = os.path.join(_models_root, "hunyuanFoley")
path_B = os.path.join(_models_root, "hunyuanfoley")
if os.path.isdir(path_A):
    _ensure_model_key(_primary_key, "hunyuanFoley")
    _ensure_model_key(_alt_key, "hunyuanFoley")
elif os.path.isdir(path_B):
    _ensure_model_key(_primary_key, "hunyuanfoley")
    _ensure_model_key(_alt_key, "hunyuanfoley")
else:
    # fallback: create primary pointing to the usual lower-case
    _ensure_model_key(_primary_key, "hunyuanfoley")
    _ensure_model_key(_alt_key, "hunyuanfoley")

# --------------------------------------------------------------------------------
# Helpers
# --------------------------------------------------------------------------------
def _pick_device():
    return mm.get_torch_device()

def _save_tensor_video_to_mp4(images: torch.Tensor, out_path: str, fps: int = 24):
    """images: (N,H,W,C) in [0..1] -> MP4"""
    frames = (images.detach().cpu().clamp(0,1).numpy() * 255).astype("uint8")
    imageio.mimsave(out_path, frames, fps=fps)

def _infer_local(video_path: str,
                 prompt: str,
                 model_dict,
                 cfg,
                 guidance_scale: float = 4.5,
                 num_inference_steps: int = 50):
    """Equivalent to repo's infer(): feature_process -> denoise_process"""
    visual_feats, text_feats, audio_len_in_s = feature_process(
        video_path, prompt, model_dict, cfg
    )
    audio, sample_rate = denoise_process(
        visual_feats,
        text_feats,
        audio_len_in_s,
        model_dict,
        cfg,
        guidance_scale=guidance_scale,
        num_inference_steps=num_inference_steps
    )
    if isinstance(audio, (list, tuple)):
        audio = audio[0]
    return audio, int(sample_rate)

def _to_layout(audio: torch.Tensor, for_vhs: bool) -> torch.Tensor:
    """
    Normalize waveform shape.
    - for_vhs=True  -> return (1, T, 1)  (works with VHS_VideoCombine and SaveAudio)
    - for_vhs=False -> return (C, T)     (classic Comfy 'channels, samples')
    """
    a = audio.detach().to(torch.float32)
    if a.ndim == 1:
        a = a.unsqueeze(0)  # (1,T)
    elif a.ndim == 3 and a.shape[0] == 1:
        # Try to detect (B,C,T)
        if a.shape[1] in (1,2):
            a = a.squeeze(0)  # (C,T)
        else:
            # assume (B,T,C)
            a = a.squeeze(0).movedim(1,0)  # (C,T)
    elif a.ndim == 2:
        pass  # (C,T) or (T,C)
    else:
        a = a.flatten().unsqueeze(0)

    # Ensure channels-first (C,T)
    if a.ndim == 2:
        C, T = a.shape
        if C not in (1,2) and T in (1,2):
            a = a.movedim(1,0)
        elif C not in (1,2):
            a = a.flatten().unsqueeze(0)

    if a.shape[0] > 2:
        a = a[:2]
    if a.shape[0] == 0:
        a = a.unsqueeze(0)

    if for_vhs:
        if a.shape[0] == 2:
            a = a.mean(0, keepdim=True)  # mono (1,T)
        a = a.movedim(0, 1)  # (T,1)
        a = a.unsqueeze(0)   # (1,T,1)
        return a.contiguous()
    else:
        return a.contiguous()

# --------------------------------------------------------------------------------
# Nodes (MMAudio-style)
# --------------------------------------------------------------------------------

class HunyuanFoleyLoader:
    """
    PICK FILES (like MMAudio):
      - MAIN  : hunyuanvideo_foley.pth
      - VAE   : vae_128d_48k.pth
      - SYNCH : synchformer_state_dict.pth
    And TYPE the YAML path (default 'config.yaml').
    All four must live in the SAME folder (e.g. models/hunyuanFoley/HunyuanVideo-Foley).
    """

    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "main_model_pth": (
                    folder_paths.get_filename_list(_primary_key),
                    {"tooltip": "Pick MAIN: hunyuanvideo_foley.pth"}
                ),
                "vae_pth": (
                    folder_paths.get_filename_list(_primary_key),
                    {"tooltip": "Pick VAE: vae_128d_48k.pth"}
                ),
                "synch_pth": (
                    folder_paths.get_filename_list(_primary_key),
                    {"tooltip": "Pick SYNCH: synchformer_state_dict.pth"}
                ),
                "config_yaml_path": (
                    "STRING",
                    {"default": "config.yaml",
                     "multiline": False,
                     "tooltip": "Absolute path, or relative to models/hunyuanFoley, or just 'config.yaml' if it sits next to the weights."}
                ),
            }
        }

    RETURN_TYPES = ("HUNYUAN_PATHS",)
    RETURN_NAMES  = ("paths",)
    FUNCTION = "pick"
    CATEGORY = "Hunyuan Foley"

    def _resolve_from_key(self, selection: str) -> str:
        # Try primary key; if not found there, try alt key
        try:
            return folder_paths.get_full_path_or_raise(_primary_key, selection)
        except Exception:
            return folder_paths.get_full_path_or_raise(_alt_key, selection)

    def pick(self, main_model_pth, vae_pth, synch_pth, config_yaml_path):
        main_path  = self._resolve_from_key(main_model_pth)
        vae_path   = self._resolve_from_key(vae_pth)
        synch_path = self._resolve_from_key(synch_pth)

        for p, name in [(main_path, "MAIN"), (vae_path, "VAE"), (synch_path, "SYNCH")]:
            if not os.path.exists(p):
                raise FileNotFoundError(f"{name} file not found: {p}")

        model_dir = os.path.dirname(main_path)
        if os.path.dirname(vae_path) != model_dir or os.path.dirname(synch_path) != model_dir:
            raise ValueError(
                "VAE and Synchformer must be in the SAME folder as the MAIN model.\n"
                f"Main dir: {model_dir}\nVAE dir: {os.path.dirname(vae_path)}\nSynch dir: {os.path.dirname(synch_path)}"
            )

        # Resolve YAML
        yaml_candidate = config_yaml_path.strip()
        if not os.path.isabs(yaml_candidate):
            # Try relative to primary key root
            try:
                yaml_from_root = folder_paths.get_full_path_or_raise(_primary_key, yaml_candidate)
            except Exception:
                yaml_from_root = None
            if yaml_from_root and os.path.exists(yaml_from_root):
                yaml_path = yaml_from_root
            else:
                yaml_path = os.path.join(model_dir, yaml_candidate)
        else:
            yaml_path = yaml_candidate

        if not os.path.exists(yaml_path):
            raise FileNotFoundError(f"Config YAML not found at: {yaml_path}")

        payload = {
            "model_dir": model_dir,
            "yaml": yaml_path,
            "main": main_path,
            "vae": vae_path,
            "synch": synch_path,
        }
        return (payload,)


class HunyuanFoleyUtilsLoader:
    """
    LOAD the models using the paths from HunyuanFoleyLoader.
    """

    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "paths": ("HUNYUAN_PATHS",),
                "precision": (["fp16", "bf16", "fp32"], {"default": "fp16"}),
            }
        }

    RETURN_TYPES = ("HUNYUAN_MODEL", "HUNYUAN_CFG")
    RETURN_NAMES  = ("model_dict", "cfg")
    FUNCTION = "loadmodel"
    CATEGORY = "Hunyuan Foley"

    def loadmodel(self, paths, precision):
        device = _pick_device()
        model_dir = paths["model_dir"]
        yaml_path = paths["yaml"]

        print(f"[HunyuanFoley] Using model folder: {model_dir}")
        print(f"[HunyuanFoley] Using config YAML : {yaml_path}")

        model_dict, cfg = load_model(model_dir, yaml_path, device)
        try:
            cfg["__precision__"] = precision
        except Exception:
            pass
        return (model_dict, cfg)


class HunyuanFoleySampler:
    """
    Generate audio from a video file path (STRING) or IMAGE frames.
    By default returns audio as (1, T, 1) to satisfy VHS_VideoCombine/SaveAudio.
    """

    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "model_dict": ("HUNYUAN_MODEL",),
                "cfg": ("HUNYUAN_CFG",),
                "prompt": ("STRING", {"multiline": True, "default": ""}),
                "guidance_scale": ("FLOAT", {"default": 4.5, "min": 0.0, "max": 20.0, "step": 0.1}),
                "num_inference_steps": ("INT", {"default": 50, "min": 1, "max": 200, "step": 1}),
                "seed": ("INT", {"default": 1, "min": 0, "max": 0xffffffffffffffff}),
                "output_for_vhs": ("BOOLEAN", {"default": True, "tooltip": "True -> (1,T,1); False -> (C,T)"}),
            },
            "optional": {
                "video_path": ("STRING", {"default": "", "tooltip": "Absolute path or models-relative path"}),
                "images": ("IMAGE",),
                "fps": ("INT", {"default": 24, "min": 1, "max": 60, "step": 1, "tooltip": "FPS for images->mp4"}),
            },
        }

    RETURN_TYPES = ("AUDIO",)
    RETURN_NAMES  = ("audio",)
    FUNCTION = "sample"
    CATEGORY = "Hunyuan Foley"

    def sample(self,
               model_dict,
               cfg,
               prompt,
               guidance_scale,
               num_inference_steps,
               seed,
               output_for_vhs,
               video_path: str = "",
               images: Optional[torch.Tensor] = None,
               fps: int = 24):

        import random
        random.seed(seed)
        np.random.seed(seed)
        torch.manual_seed(seed)

        tmp_mp4 = None
        try:
            if images is not None:
                tmp_dir = tempfile.mkdtemp(prefix="hunyuanfoley_")
                tmp_mp4 = os.path.join(tmp_dir, "input.mp4")
                _save_tensor_video_to_mp4(images, tmp_mp4, fps=fps)
                vid_in = tmp_mp4
            else:
                if not video_path:
                    raise ValueError("Provide either 'video_path' or 'images'.")
                try:
                    vid_in = folder_paths.get_full_path_or_raise(_primary_key, video_path)
                except Exception:
                    try:
                        vid_in = folder_paths.get_full_path_or_raise(_alt_key, video_path)
                    except Exception:
                        vid_in = video_path
                if not os.path.exists(vid_in):
                    raise FileNotFoundError(f"Video not found: {vid_in}")

            waveform, sample_rate = _infer_local(
                vid_in, prompt, model_dict, cfg,
                guidance_scale=guidance_scale,
                num_inference_steps=num_inference_steps
            )

            waveform = _to_layout(waveform, for_vhs=output_for_vhs)
            audio = {"waveform": waveform.cpu(), "sample_rate": int(sample_rate)}
            return (audio,)

        finally:
            if tmp_mp4 is not None:
                try:
                    d = os.path.dirname(tmp_mp4)
                    if os.path.exists(tmp_mp4):
                        os.remove(tmp_mp4)
                    if os.path.isdir(d):
                        os.rmdir(d)
                except Exception:
                    pass


# --------------------------------------------------------------------------------
# Registry
# --------------------------------------------------------------------------------
NODE_CLASS_MAPPINGS = {
    "HunyuanFoleyLoader":       HunyuanFoleyLoader,
    "HunyuanFoleyUtilsLoader":  HunyuanFoleyUtilsLoader,
    "HunyuanFoleySampler":      HunyuanFoleySampler,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "HunyuanFoleyLoader":      "Hunyuan Foley - File Picker",
    "HunyuanFoleyUtilsLoader": "Hunyuan Foley - Model Loader",
    "HunyuanFoleySampler":     "Hunyuan Foley - Sampler",
}
