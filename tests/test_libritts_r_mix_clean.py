import itertools

import numpy as np

from lhotse_dataset.libritts_r_mix_clean import LibriTTSRMixClean


def test_libritts_r() -> None:
    corpus = LibriTTSRMixClean()
    gen = corpus.get_cuts()
    for cut in itertools.islice(gen, 3):
        audio = cut.load_audio()
        assert isinstance(audio, np.ndarray)
        assert cut.sampling_rate == 24_000
        assert cut.duration == audio.shape[1] / cut.sampling_rate
