import sys, re
from torch import no_grad, LongTensor
import logging

logging.getLogger('numba').setLevel(logging.WARNING)

import commons
import utils
from models import SynthesizerTrn
from text import text_to_sequence, _clean_text
from mel_processing import spectrogram_torch

from scipy.io.wavfile import write

def get_text(text, hps, cleaned=False):
    if cleaned:
        text_norm = text_to_sequence(text, hps.symbols, [])
    else:
        text_norm = text_to_sequence(text, hps.symbols, hps.data.text_cleaners)
    if hps.data.add_blank:
        text_norm = commons.intersperse(text_norm, 0)
    text_norm = LongTensor(text_norm)
    return text_norm

def ask_if_continue():
    while True:
        answer = input('Continue? (y/n): ')
        if answer == 'y':
            break
        elif answer == 'n':
            sys.exit(0)

def print_speakers(speakers):
    print('ID\tSpeaker')
    for id, name in enumerate(speakers):
        print(str(id) + '\t' + name)

def get_speaker_id(message):
    speaker_id = input(message)
    try:
        speaker_id = int(speaker_id)
    except:
        print(str(speaker_id) + ' is not a valid ID!')
        sys.exit(1)
    return speaker_id

def get_label_value(text, label, default, warning_name='value'):
    value=re.search(rf'\[{label}=(.+?)\]',text)
    if value:
        try:
            text=re.sub(rf'\[{label}=(.+?)\]','',text,1)
            value=float(value.group(1))
        except:
            print(f'Invalid {warning_name}!')
            sys.exit(1)
    else:
        value=default
    return value, text

def get_label(text,label):
    if f'[{label}]' in text:
        return True,text.replace(f'[{label}]','')
    else:
        return False,text

if __name__ == '__main__':
    model = input('Path of a VITS model: ')
    config = input('Path of a config file: ')
    try:
        hps_ms = utils.get_hparams_from_file(config)
        if 'n_speakers' in hps_ms.data.keys():
            n_speakers=hps_ms.data.n_speakers
        else:
            n_speakers=0
        if 'symbols' in hps_ms.keys():
            n_symbols=len(hps_ms.symbols)
        else:
            n_symbols=0
        if 'speakers' in hps_ms.keys():
            speakers=hps_ms.speakers
        elif n_speakers>0:
            speakers=[str(i) for i in range(n_speakers)]
        else:
            speakers=['0']
        net_g_ms = SynthesizerTrn(
            n_symbols,
            hps_ms.data.filter_length // 2 + 1,
            hps_ms.train.segment_size // hps_ms.data.hop_length,
            n_speakers=n_speakers,
            **hps_ms.model)
        _ = net_g_ms.eval()
        utils.load_checkpoint(model, net_g_ms)
    except:
        print('Failed to load!')
        sys.exit(1)
    
    if n_symbols!=0:
        while True:
            choice = input('TTS or VC? (t/v):')
            if choice == 't':
                text = input('Text to read: ')
                if text=='[ADVANCED]':
                    text = input('Raw text:')
                    print('Cleaned text is:')
                    print(_clean_text(text, hps_ms.data.text_cleaners))
                    continue
                
                length_scale,text=get_label_value(text,'LENGTH',1,'length scale')
                cleaned,text=get_label(text,'CLEANED')
                try:
                    stn_tst = get_text(text, hps_ms, cleaned=cleaned)
                except:
                    print('Invalid text!')
                    sys.exit(1)
                
                print_speakers(speakers)
                speaker_id = get_speaker_id('Speaker ID: ')
                out_path = input('Path to save: ')

                try:
                    with no_grad():
                        x_tst = stn_tst.unsqueeze(0)
                        x_tst_lengths = LongTensor([stn_tst.size(0)])
                        sid = LongTensor([speaker_id])
                        audio = net_g_ms.infer(x_tst, x_tst_lengths, sid=sid, noise_scale=.667, noise_scale_w=0.8, length_scale=length_scale)[0][0,0].data.cpu().float().numpy()
                    write(out_path, hps_ms.data.sampling_rate, audio)
                except:
                    print('Failed to generate!')
                    sys.exit(1)
                
                print('Successfully saved!')
                ask_if_continue()
                
                    
            elif choice == 'v':
                audio_path = input('Path of an audio file to convert:\n')
                print_speakers(speakers)
                audio = utils.load_audio_to_torch(audio_path, hps_ms.data.sampling_rate)

                originnal_id = get_speaker_id('Original speaker ID: ')
                target_id = get_speaker_id('Target speaker ID: ')
                out_path = input('Path to save: ')

                y = audio.unsqueeze(0)

                spec = spectrogram_torch(y, hps_ms.data.filter_length,
                    hps_ms.data.sampling_rate, hps_ms.data.hop_length, hps_ms.data.win_length,
                    center=False)
                spec_lengths = LongTensor([spec.size(-1)])
                sid_src = LongTensor([originnal_id])

                try:
                    with no_grad():
                        sid_tgt = LongTensor([target_id])
                        audio = net_g_ms.voice_conversion(spec, spec_lengths, sid_src=sid_src, sid_tgt=sid_tgt)[0][0,0].data.cpu().float().numpy()
                    write(out_path, hps_ms.data.sampling_rate, audio)
                except:
                    print('Failed to generate!')
                    sys.exit(1)
                        
                print('Successfully saved!')
                ask_if_continue()
    else:
        model = input('Path of a hubert-soft model: ')
        from hubert_model import hubert_soft
        try:
            hubert = hubert_soft(model)
        except:
            print('Failed to load!')
            sys.exit(1)
        while True:
            audio_path = input('Path of an audio file to convert:\n')
            print_speakers(speakers)
            audio = utils.load_audio_to_torch(audio_path, 16000).unsqueeze(0).unsqueeze(0)

            target_id = get_speaker_id('Target speaker ID: ')
            out_path = input('Path to save: ')
            length_scale,out_path=get_label_value(out_path,'LENGTH',1,'length scale')

            from torch import inference_mode
            try:
                with inference_mode():
                    unit = hubert.units(audio)
                    unit_lengths = LongTensor([unit.size(1)])
                    sid = LongTensor([target_id])
                    audio = net_g_ms.infer(unit, unit_lengths, sid=sid, noise_scale=.667, noise_scale_w=0.8, length_scale=length_scale)[0][0,0].data.float().numpy()
                write(out_path, hps_ms.data.sampling_rate, audio)
            except:
                print('Failed to generate!')
                sys.exit(1)
            
            print('Successfully saved!')
            ask_if_continue()
