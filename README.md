# ComfyUI_HunyuanFoley
ComfyUI wrapper for **Tencent HunyuanVideo-Foley**.  

Workflow: created by **[https://aistudynow.com/hunyuanvideo-foley-comfyui-workflow-turn-quiet-video-into-sound/](https://aistudynow.com/hunyuanvideo-foley-comfyui-workflow-turn-quiet-video-into-sound/)**

## Tutorial Video

[![ComfyUI HunyuanVideo-Foley Tutorial](https://img.youtube.com/vi/TpxkErTzawg/0.jpg)](https://www.youtube.com/watch?v=TpxkErTzawg)


---

## Models

Get the files from the official release: **[HunyuanVideo-Foley on Hugging Face](https://huggingface.co/tencent/HunyuanVideo-Foley/tree/main)**

Place all HunyuanVideo-Foley weights and `config.yaml` in one folder (either `hunyuanFoley` or `hunyuanfoley` is accepted):

```text
ComfyUI/models/hunyuanFoley/HunyuanVideo-Foley/
├─ hunyuanvideo_foley.pth
├─ vae_128d_48k.pth
├─ synchformer_state_dict.pth
└─ config.yaml

