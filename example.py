# Load lib
import tts

# Load tts model
tts.load('D:/PROJECT/0_py/moe_tts/model/MoeGoe/1164_epochs.pth', 
          'D:/PROJECT/0_py/moe_tts/model/MoeGoe/config.json')

# Generate wav file
tts.wav('이것은 테스트 문장입니다.')

# You can change speaker and path
tts.wav(text='이것은 테스트 문장입니다.', speaker_id=0, path='./demo.wav')


# You can receive the data as array format
## it returns Numpy array
data = tts.main('이것은 테스트 문장입니다.', 5) # this model has 6 speakers(0~5)

## And you can play directly from this
import simpleaudio as sa
sampling_rate =  tts.hps_ms.data.sampling_rate

sa_obj = sa.play_buffer(data, 1, 4, sampling_rate)
sa_obj.wait_done()