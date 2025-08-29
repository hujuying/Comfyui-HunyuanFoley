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


Windows portable install:

# run this from ComfyUI_windows_portable folder
python_embeded\python.exe -m pip install -r ComfyUI\custom_nodes\ComfyUI_HunyuanFoley\requirements.txt


Minimal extras required by this node: imageio, imageio-ffmpeg, numpy.
PyTorch & the rest come from ComfyUI and the bundled Hunyuan code.

Restart ComfyUI.

Models

Place all HunyuanVideo-Foley weights and config.yaml in a single folder:

ComfyUI/models/hunyuanFoley/HunyuanVideo-Foley/
├─ hunyuanvideo_foley.pth
├─ vae_128d_48k.pth
├─ synchformer_state_dict.pth
└─ config.yaml


Notes:

The folder name can be hunyuanFoley or hunyuanfoley — this node accepts either.

All four files must live together in the same directory.

Get the files from the official HunyuanVideo-Foley release (Tencent/author distribution).
