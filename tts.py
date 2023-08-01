import sys
from scipy.io.wavfile import write
from mel_processing import spectrogram_torch
from text import text_to_sequence, _clean_text
from models import SynthesizerTrn
import utils
import commons
import re
from torch import no_grad, LongTensor

def get_label_value(text, label, default, warning_name='value'):
    value = re.search(rf'\[{label}=(.+?)\]', text)
    if value:
        try:
            text = re.sub(rf'\[{label}=(.+?)\]', '', text, 1)
            value = float(value.group(1))
        except:
            print(f'Invalid {warning_name}!')
            sys.exit(1)
    else:
        value = default
    return value, text

def get_text(text, hps, cleaned=False):
    if cleaned:
        text_norm = text_to_sequence(text, hps.symbols, [])
    else:
        text_norm = text_to_sequence(text, hps.symbols, hps.data.text_cleaners)
    if hps.data.add_blank:
        text_norm = commons.intersperse(text_norm, 0)
    text_norm = LongTensor(text_norm)
    return text_norm

def get_label(text, label):
    if f'[{label}]' in text:
        return True, text.replace(f'[{label}]', '')
    else:
        return False, text



net_g_ms, hps_ms = None, None
def load(model:str, config:str):
    global net_g_ms, hps_ms
    
    hps_ms = utils.get_hparams_from_file(config)

    n_speakers = hps_ms.data.n_speakers if 'n_speakers' in hps_ms.data.keys() else 0
    n_symbols = len(hps_ms.symbols) if 'symbols' in hps_ms.keys() else 0
    speakers = hps_ms.speakers if 'speakers' in hps_ms.keys() else ['0']
    use_f0 = hps_ms.data.use_f0 if 'use_f0' in hps_ms.data.keys() else False
    emotion_embedding = hps_ms.data.emotion_embedding if 'emotion_embedding' in hps_ms.data.keys() else False
    
    net_g_ms = SynthesizerTrn(
        n_symbols,
        hps_ms.data.filter_length // 2 + 1,
        hps_ms.train.segment_size // hps_ms.data.hop_length,
        n_speakers=n_speakers,
        emotion_embedding=emotion_embedding,
        **hps_ms.model)
    _ = net_g_ms.eval()

    utils.load_checkpoint(model, net_g_ms)
    print('TTS successfully loaded!')


def wav(text:str, speaker_id:int=0, path:str='./demo.wav'):
    try:
        write(path, hps_ms.data.sampling_rate, main(text, speaker_id))
    except PermissionError as e:
        print(e)
        print('|ALERT| {path} should include file name!')
        write(path+'/demo.wav', hps_ms.data.sampling_rate, main(text, speaker_id))
        
    print('Successfully saved!')
    return path


def main(text:str, speaker_id:int=0):
    length_scale, text = get_label_value(
        text, 'LENGTH', 1, 'length scale')
    noise_scale, text = get_label_value(
        text, 'NOISE', 0.667, 'noise scale')
    noise_scale_w, text = get_label_value(
        text, 'NOISEW', 0.8, 'deviation of noise')
    cleaned, text = get_label(text, 'CLEANED')

    stn_tst = get_text(text, hps_ms, cleaned=cleaned)
    
    with no_grad():
        x_tst = stn_tst.unsqueeze(0)
        x_tst_lengths = LongTensor([stn_tst.size(0)])
        sid = LongTensor([speaker_id])
        audio = net_g_ms.infer(x_tst, x_tst_lengths, sid=sid, noise_scale=noise_scale,
                                noise_scale_w=noise_scale_w, length_scale=length_scale)[0][0, 0].data.cpu().float().numpy()
    
    return audio


if __name__ == '__main__':
    load()
    wav('틀이모양을 이루지못하고 천초백식을', 0)