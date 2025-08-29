# ComfyUI_HunyuanFoley
ComfyUI wrapper for **Tencent HunyuanVideo-Foley**.  
Pick MAIN / VAE / SYNCH files explicitly and type a `config.yaml` path.  
Outputs audio shaped for either **VHS_VideoCombine** or **Save/PreviewAudio**.

---

## Models

Place all HunyuanVideo-Foley weights and `config.yaml` in one folder (either `hunyuanFoley` or `hunyuanfoley` is accepted):

```text
ComfyUI/models/hunyuanFoley/HunyuanVideo-Foley/
├─ hunyuanvideo_foley.pth
├─ vae_128d_48k.pth
├─ synchformer_state_dict.pth
└─ config.yaml
