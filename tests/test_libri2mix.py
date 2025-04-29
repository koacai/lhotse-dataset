import itertools
from pathlib import Path

import numpy as np

from lhotse_dataset.libri2mix import Libri2Mix


def test_path_to_librispeech_id() -> None:
    path = "dev-clean/1462/170142/1462-170142-0040.flac"
    id = Libri2Mix.path_to_librispeech_id(path)
    assert id == "librispeech_dev-clean_1462-170142-0040"


def test_librimix() -> None:
    librispeech_shar_dir = Path("/groups/gag51394/users/asai/shar/librispeech")
    corpus = Libri2Mix(librispeech_shar_dir)
    gen = corpus.get_cuts()
    for cut in itertools.islice(gen, 3):
        audio = cut.load_audio()
        assert isinstance(audio, np.ndarray)
        assert cut.sampling_rate == 16_000
        assert cut.duration == audio.shape[1] / cut.sampling_rate
