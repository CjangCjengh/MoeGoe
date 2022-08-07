# GUI
See [MoeGoe_GUI](https://github.com/CjangCjengh/MoeGoe_GUI)
# Models
## Nene + Meguru + Yoshino + Mako + Murasame + Koharu + Nanami
Download [Config File](https://sjtueducn-my.sharepoint.com/:u:/g/personal/cjang_cjengh_sjtu_edu_cn/Ed7PXqaBdllAki0TPpeZorgBFdnxirbX_AYGUIiIcWAYNg?e=avxkWs)

Download [Model](https://sjtueducn-my.sharepoint.com/:u:/g/personal/cjang_cjengh_sjtu_edu_cn/EQ_X9rsRd6tCrztZcQ_ad6QBz8GNSnpPq9H_C6ASoLDkfA?e=LZHD2O) (243 epochs)
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
