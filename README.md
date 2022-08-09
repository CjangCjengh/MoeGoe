# GUI
See [MoeGoe_GUI](https://github.com/CjangCjengh/MoeGoe_GUI)
# Online demo
Integrated into [Huggingface Spaces 🤗](https://huggingface.co/spaces) using [Gradio](https://github.com/gradio-app/gradio). Try it out [![Hugging Face Spaces](https://img.shields.io/badge/%F0%9F%A4%97%20Hugging%20Face-Spaces-blue)](https://huggingface.co/spaces/skytnt/moe-japanese-tts)
# Models
## Nene + Meguru + Yoshino + Mako + Murasame + Koharu + Nanami
Download [Config File](https://sjtueducn-my.sharepoint.com/:u:/g/personal/cjang_cjengh_sjtu_edu_cn/Ed7PXqaBdllAki0TPpeZorgBFdnxirbX_AYGUIiIcWAYNg?e=avxkWs)

Download [Model](https://sjtueducn-my.sharepoint.com/:u:/g/personal/cjang_cjengh_sjtu_edu_cn/EW0gHn46FeRGsoGZvWTy-FwBrKFhzKYf3n9_vLFeNoz_FA?e=gpkjED) (365 epochs)
# How to use
Run MoeGoe.exe
```
Path of a VITS model: D:\Download\243_epochs.pth
Path of a config file: D:\Download\config.json
INFO:root:Loaded checkpoint 'D:\Download\243_epochs.pth' (iteration 243)
```
## Text to speech
```
TTS or VC? (t/v):t
Text to read: こんにちは。
ID      Speaker
0       綾地寧々
1       因幡めぐる
2       朝武芳乃
3       常陸茉子
4       ムラサメ
5       鞍馬小春
6       在原七海
Speaker ID: 0
Path to save: demo.wav
Successfully saved!
```
## Voice conversion
```
TTS or VC? (t/v):v
Path of a WAV file (22050 Hz, 16 bits, 1 channel) to convert:
D:\dataset\ayachi_nene\nen001_001.wav
ID      Speaker
0       綾地寧々
1       因幡めぐる
2       朝武芳乃
3       常陸茉子
4       ムラサメ
5       鞍馬小春
6       在原七海
Original speaker ID: 0
Target speaker ID: 6
Path to save: demo.wav
Successfully saved!
```
