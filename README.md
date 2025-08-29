# ComfyUI_HunyuanFoley
ComfyUI wrapper for **Tencent HunyuanVideo-Foley**.  
Pick MAIN / VAE / SYNCH files explicitly and type a `config.yaml` path.  
Outputs audio shaped for either **VHS_VideoCombine** or **Save/PreviewAudio**.

---

## Installation

Clone this repo into your `custom_nodes` folder.

```bash
# from your ComfyUI root
git clone https://github.com/yourname/ComfyUI_HunyuanFoley ComfyUI/custom_nodes/ComfyUI_HunyuanFoley

Install Python dependencies:

# standard Python install
pip install -r ComfyUI/custom_nodes/ComfyUI_HunyuanFoley/requirements.txt


Nodes Overview

Hunyuan Foley – File Picker
Lets you select:

main_model_pth → hunyuanvideo_foley.pth

vae_pth → vae_128d_48k.pth

synch_pth → synchformer_state_dict.pth

config_yaml_path → Path to config.yaml (absolute path, or relative to your models folder; default config.yaml if it sits next to the weights)

Output: paths (internal dict of resolved file paths)

Hunyuan Foley – Model Loader
Loads all models using the paths from File Picker.
Outputs:

model_dict (HUNYUAN_MODEL)

cfg (HUNYUAN_CFG)

Hunyuan Foley – Sampler
Generates audio from a video + prompt.
Inputs:

model_dict, cfg (from Model Loader)

prompt (text)

guidance_scale, num_inference_steps, seed

video_path or images (if images, the node encodes a temp MP4 automatically)

output_layout:

VHS → outputs shape (1, 1, T) for VHS_VideoCombine

COMFY → outputs shape (1, T) for SaveAudio / PreviewAudio
Output: audio
