# Links
- [MoeGoe_GUI](https://github.com/CjangCjengh/MoeGoe_GUI)
- [Pretrained models](https://github.com/CjangCjengh/TTSModels)

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
0       XXXX
1       XXXX
2       XXXX
Speaker ID: 0
Path to save: demo.wav
Successfully saved!
```
## Voice conversion
```
TTS or VC? (t/v):v
Path of an audio file to convert:
D:\dataset\demo.wav
ID      Speaker
0       XXXX
1       XXXX
2       XXXX
Original speaker ID: 0
Target speaker ID: 6
Path to save: demo.wav
Successfully saved!
```
## HuBERT-VITS
```
Path of a hubert-soft model: D:\Download\hubert-soft.pt
Path of an audio file to convert:
D:\dataset\demo.wav
ID      Speaker
0       XXXX
1       XXXX
2       XXXX
Target speaker ID: 6
Path to save: demo.wav
Successfully saved!
```
