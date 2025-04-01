import itertools

import numpy as np

from lhotse_dataset.callhome_jp import CallHomeJP


class TestCallHomeJP:
    def test_get_cuts(self):
        email = "email"  # NEED TO CHANGE
        pswd = "pswd"  # NEED TO CHANGE
        corpus = CallHomeJP(email, pswd)
        gen = corpus.get_cuts()
        for cut in itertools.islice(gen, 3):
            audio = cut.load_audio()
            assert isinstance(audio, np.ndarray)
            assert cut.sampling_rate == 8_000
            assert cut.duration == audio.shape[1] / cut.sampling_rate
