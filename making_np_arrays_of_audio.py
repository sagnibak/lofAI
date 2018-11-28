# coding: utf-8
from pydub import AudioSegment
dd = r'D:\sagniD\audio_data\lofi\lofi0.mp3'
dd
song = AudioSegment.from_mp3(dd)
song = AudioSegment.from_mp3(dd)
song[:20]
song[:20]
song[:20]
song[:]
song[:20]
song[:20]
song[:20]
song[:20]
from pydub.playback import play
# play(song[:4000])
# import simpleaudio
# play(song[:4000])
# play(song[:4000])
# play(song[:12000])
s16k = song.set_frame_rate(16000)
# play(s16k[:12000])
# play(s16k[120000:120000 + 20000])
# play(s16k[120000:120000 + 30000])
# play(s16k[120000:120000 + 60000])
index = 0

def song_to_np_array(song):
    return np.array(song.get_array_of_samples())

# song_to_np_array(song[:25])
import numpy as np
song_to_np_array(song[:25])
song_to_np_array(song[1000:1025])
len(song_to_np_array(song))
len(song_to_np_array(s16k))
arr16k = song_to_np_array(s16k)
arr16k.shape
arr16k.dtype
arr16k = arr16k.reshape(-1, 2)
arr16k.shape
arr16k
arr16k = arr16k.T
arr16k
arr16k[:, 1000:1025]
arr16k[:, 16000:16400]
globals()
locals()
# max(arr16k)
np.max(arr16k)
# np.maximum(arr16k)
np.max(arr16k)
np.min(arr16k)
arr16k = arr16k.astype(np.float32)
np.max(arr16k)
np.min(arr16k)
arr16k /= 32768.
np.min(arr16k)
np.max(arr16k)
arr16k
arr16k[:, 16000:16100]
arr16k[:, 16000:16100].shape
arr16k[0, 16000:16100].shape
arr16k[1, 16000:16100].shape
# get_ipython().system('start .')
# get_ipython().system('mkdir audio_training')
# get_ipython().system('ls')
# np.save('audio_training/lofi_left.npy', arr16k[0])
# np.save('audio_training/lofi_right.npy', arr16k[1])
arr16k[0].shape
arr16k[1].shape
channels_last = arr16k.T
channels_last.shape
np.save('audio_training/channels_last.npy', channels_last)
