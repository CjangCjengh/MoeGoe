import sys
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

if __name__ == '__main__':
    model = input('Path of a VITS model: ')
    config = input('Path of a config file: ')
    try:
        hps_ms = utils.get_hparams_from_file(config)
        net_g_ms = SynthesizerTrn(
            len(hps_ms.symbols),
            hps_ms.data.filter_length // 2 + 1,
            hps_ms.train.segment_size // hps_ms.data.hop_length,
            n_speakers=hps_ms.data.n_speakers,
            **hps_ms.model)
        _ = net_g_ms.eval()
        _ = utils.load_checkpoint(model, net_g_ms, None)
    except:
        print('Failed to load!')
        sys.exit(1)

    while True:
        choice = input('TTS or VC? (t/v):')
        if choice == 't':
            text = input('Text to read: ')
            if text=='[ADVANCED]':
                text = input('Raw text:')
                print('Cleaned text is:')
                print(_clean_text(text, hps_ms.data.text_cleaners))
                text = input('Cleaned text to read:\n')
                try:
                    stn_tst = get_text(text, hps_ms, cleaned=True)
                except:
                    print('Invalid text!')
                    sys.exit(1)
            else:
                try:
                    stn_tst = get_text(text, hps_ms)
                except:
                    print('Invalid text!')
                    sys.exit(1)
            
            print_speakers(hps_ms.speakers)
            speaker_id = get_speaker_id('Speaker ID: ')

            out_path = input('Path to save: ')

            try:
                with no_grad():
                    x_tst = stn_tst.unsqueeze(0)
                    x_tst_lengths = LongTensor([stn_tst.size(0)])
                    sid = LongTensor([speaker_id])
                    audio = net_g_ms.infer(x_tst, x_tst_lengths, sid=sid, noise_scale=.667, noise_scale_w=0.8, length_scale=1)[0][0,0].data.cpu().float().numpy()
                write(out_path, hps_ms.data.sampling_rate, audio)
            except:
                print('Failed to generate!')
                sys.exit(1)
            
            print('Successfully saved!')
            ask_if_continue()
            
                
        elif choice == 'v':
            audio_path = input('Path of an audio file to convert:\n')
            print_speakers(hps_ms.speakers)
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