import itertools

import numpy as np

from lhotse_dataset.hq_youtube import HQYouTube


class TestHQYouTube:
    def test_get_cuts(self):
        corpus = HQYouTube("/path/to/hq-youtube.tar")
        gen = corpus.get_cuts()
        for cut in itertools.islice(gen, 3):
            audio = cut.load_audio()
            assert isinstance(audio, np.ndarray)
            assert cut.sampling_rate == 24_000
            assert cut.duration == audio.shape[1] / cut.sampling_rate
