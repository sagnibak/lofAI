import numpy as np


# mu-law companding
def mu_law_compression(x, mu=255):
    """Function to perform mu-law compression, as done in the og WaveNet."""
    return np.sign(x) * np.log(1 + mu*np.absolute(x)) / np.log(1 + mu)

def mu_law_expansion(x, mu=255):
    """Function to perform mu-law expansion, the inverse of mu-law compression."""
    return np.sign(x) * (np.power(1 + mu, np.absolute(x)) - 1) / mu


try:
    mu_law_audio = np.load('audio_training/mu_law_compressed.npy')
except FileNotFoundError as e:
    mu_law_audio = None
raw_audio = None  # load raw audio only if necessary (lazily)

def get_segment(start_idx: int, num_samples: int=256_000) -> np.ndarray:
    """Returns a Numpy ndarray of type np.float32 of shape (`num_samples`, 2),
    containing sound signals with mu-law compression applied.
    """
    return mu_law_audio[start_idx: start_idx + num_samples]


def get_raw_segment(start_idx: int, num_samples: int=256_000):
    """Returns a Numpy ndarray of type np.float32 of shape (`num_samples`, 2),
    containing raw sound signals.
    """
    global raw_audio
    if raw_audio is None:
        raw_audio = np.load('audio_training/channels_last.npy')
    return raw_audio[start_idx: start_idx + num_samples]
