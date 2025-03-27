import itertools

import numpy as np

from lhotse_dataset.jsut import JSUT


class TestJSUT:
    def test_get_cuts(self):
        corpus = JSUT()
        gen = corpus.get_cuts()
        for cut in itertools.islice(gen, 3):
            audio = cut.load_audio()
            assert isinstance(audio, np.ndarray)
            assert cut.sampling_rate == 48_000
            assert cut.duration == audio.shape[1] / cut.sampling_rate
