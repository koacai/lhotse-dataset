import itertools

import numpy as np

from lhotse_dataset.librispeech import LibriSpeech


class TestLibriSpeech:
    def test_get_cuts(self):
        corpus = LibriSpeech()
        gen = corpus.get_cuts()
        for cut in itertools.islice(gen, 3):
            audio = cut.load_audio()
            assert isinstance(audio, np.ndarray)
            assert cut.sampling_rate == 16_000
            assert cut.duration == audio.shape[1] / cut.sampling_rate
