from math import floor
import numpy as np
from keras.utils import Sequence
from random import randint


# mu-law companding
def mu_law_compression(x, mu=255):
    """Function to perform mu-law compression, as done in the og WaveNet."""
    return np.sign(x) * np.log(1 + mu*np.absolute(x)) / np.log(1 + mu)

def mu_law_expansion(x, mu=255):
    """Function to perform mu-law expansion, the inverse of mu-law compression."""
    return np.sign(x) * (np.power(1 + mu, np.absolute(x)) - 1) / mu


try:
    mdct_mu_law_stereo = np.load('audio_training/mdct_mu_lofi_stereo.npy')
    mu_law_audio = np.load('audio_training/mu_law_compressed.npy')
except FileNotFoundError as e:
    mu_law_audio = None
#     mdct_mu_law_stereo = None
raw_audio = None  # load raw audio only if necessary (lazily)


def get_segment(start_idx: int, num_samples: int=1024) -> np.ndarray:
    """Returns a Numpy ndarray of type np.float32 of shape (`num_samples`, 2),
    containing sound signals with mu-law compression applied.
    """
    return mdct_mu_law_stereo[:, start_idx: start_idx + num_samples]


def get_raw_segment(start_idx: int, num_samples: int=256_000):
    """Returns a Numpy ndarray of type np.float32 of shape (`num_samples`, 2),
    containing raw sound signals.
    """
    global raw_audio
    if raw_audio is None:
        raw_audio = np.load('audio_training/channels_last.npy')
    return raw_audio[start_idx: start_idx + num_samples]


class LofiSequence(Sequence):

    def __init__(self, batch_size, validation=False):
        self.batch_size = batch_size
        self.validation = validation

    def actual_length(self):
        return 2 * floor(mdct_mu_law_stereo.shape[1] / (self.batch_size * 1024))

    def __len__(self):
        if not self.validation:
            return self.actual_length()
        return 15

    def __getitem__(self, idx):
        if not self.validation:
            x = np.array([get_segment(start_idx=i * 512)
                          for i in range(idx * self.batch_size, (idx + 1) * self.batch_size)])
        else:
            random_start = randint(0, self.actual_length() - self.batch_size - 1)
            x = np.array([get_segment(start_idx=i * 512)
                          for i in range((random_start) * self.batch_size,
                                         (random_start + 1) * self.batch_size)])
        return x, x
