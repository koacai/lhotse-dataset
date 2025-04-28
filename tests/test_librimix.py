import itertools
from pathlib import Path

import numpy as np

from lhotse_dataset.librimix import LibriMix


def test_path_to_librispeech_id() -> None:
    path = "dev-clean/1462/170142/1462-170142-0040.flac"
    id = LibriMix.path_to_librispeech_id(path)
    assert id == "librispeech_dev-clean_1462-170142-0040"


def test_path_to_noise_id() -> None:
    path = "cv/20oo0109_0.72115_01ua010r_-0.72115.wav"
    id = LibriMix.path_to_noise_id(path)
    assert id == "20oo0109_0.72115_01ua010r_-0.72115"


def test_librimix() -> None:
    librispeech_shar_dir = Path("/groups/gag51394/users/asai/shar/librispeech")
    wham_noise_shar_dir = Path("/groups/gag51394/users/asai/shar/wham_noise")
    corpus = LibriMix(librispeech_shar_dir, wham_noise_shar_dir)
    gen = corpus.get_cuts()
    for cut in itertools.islice(gen, 3):
        audio = cut.load_audio()
        assert isinstance(audio, np.ndarray)
        assert cut.sampling_rate == 16_000
        assert cut.duration == audio.shape[1] / cut.sampling_rate

        for track in cut.tracks:
            c = track.cut
            audio = c.load_audio()
            assert isinstance(audio, np.ndarray)
            assert c.sampling_rate == 16_000
            assert c.duration == audio.shape[1] / cut.sampling_rate
