import itertools

import numpy as np

from lhotse_dataset.libri2mix import Libri2Mix


def test_librimix() -> None:
    corpus = Libri2Mix()
    gen = corpus.get_cuts()
    for cut in itertools.islice(gen, 3):
        audio = cut.load_audio()
        assert isinstance(audio, np.ndarray)
        assert cut.sampling_rate == 16_000
        assert cut.duration == audio.shape[1] / cut.sampling_rate
