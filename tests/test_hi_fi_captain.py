import itertools

import numpy as np

from lhotse_dataset.hi_fi_captain import HiFiCAPTAIN


class TestHiFiCAPTAIN:
    def test_get_cuts(self):
        corpus = HiFiCAPTAIN()
        gen = corpus.get_cuts()
        for cut in itertools.islice(gen, 3):
            audio = cut.load_audio()
            assert isinstance(audio, np.ndarray)
            assert cut.sampling_rate == 48_000
            assert cut.duration == audio.shape[1] / cut.sampling_rate
